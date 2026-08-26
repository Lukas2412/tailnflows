from pathlib import Path
from functools import partial

import torch

from nflows.transforms.base import Transform

from tailnflows.targets.data import real_data_sources
from tailnflows.models import flows
from tailnflows.train import data_fit
from tailnflows.utils import add_raw_data, get_experiment_output_path, load_raw_data, get_data_path, add_experiment_output_data, parallel_runner
from tailnflows.models.preprocessing import inverse_and_lad as t_to_norm_inverse_and_lad

DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)

gpu_ix = 3
if torch.cuda.is_available():
    torch.set_default_device(f"cuda:{gpu_ix}")
    DEFAULT_DEVICE = torch.device(f"cuda:{gpu_ix}")
else:
    torch.set_default_device("cpu")
    DEFAULT_DEVICE = torch.device("cpu")


# NOTE: I think this script is supposed to only run fama5, insurance and sp500 experiments (not climdex!)

"""
Model specifications
"""

def base_rqs_spec(dim: int, model_config: dict) -> list[Transform]:
    """ Builds a list of transformations representing a NSF with {depth} rational quadratic spline (rqs) layers and LU layers. """
    return flows.base_nsf_transform(
        dim, 
        num_bins=model_config.get('num_bins', 5),
        tail_bound=model_config.get('tail_bound', 3),
        affine_autoreg_layer=True,
        depth=model_config.get('depth', 1),
        u_linear_layer=True
    )

def normal(dim: int, dfs, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model without a final tail transformation. """
    return flows.build_base_model(
        dim, 
        use="density_estimation", 
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        final_rotation="lu",
    )

def get_preprocessor(dfs: list[float]): # Used for training a normal model with an initial Gaussian to Student-t transformation.
    def _preprocess(x):
        z, lad = t_to_norm_inverse_and_lad(
            x.cpu(), 
            [
                df if df > 0 else 30.
                for df in dfs
            ]
        )
        return z.to(DEFAULT_DTYPE).to(x.device), lad.to(DEFAULT_DTYPE).to(x.device)

    return _preprocess

def ttf_rqs(dim: int, dfs, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final TTF transformation with learnable tail params, initialized by sampling uniformly from [0.05, 1.0].
        As the tail param are initialized randomly, the dfs argument is non-used. """
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=[
                float(t.cpu()) for t in torch.distributions.Uniform(low=0.05, high=1.0).sample([dim])
            ],
            neg_tail_init=[
                float(t.cpu()) for t in torch.distributions.Uniform(low=0.05, high=1.0).sample([dim])
            ]
        ),
        final_rotation="lu",
    )


def ttf_rqs_fix(dim: int, dfs: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final TTF transformation with non-learnable tail params,
        initialized symmetrically for pos and neg tails given by the values in dfs. """
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=True,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in dfs['dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in dfs['dfs']],
        ),
        final_rotation="lu",
    )


def ttf_rqs_mod(dim: int, dfs: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final modified TTF transformation with tail params given by dfs.
        NOTE: Edit model_kwargs to specify TTF modifications! """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=True,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in dfs['dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in dfs['dfs']],
            hd_only=False,
            mod="qua",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=True,
        ),
        final_rotation="lu",
    )



def gtaf_rqs(dim: int, dfs, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF with a trainable Student-t base distribution with marginal dfs sampled uniformly from [1.0, 20.0] """
    return flows.build_gtaf(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            tail_init=torch.distributions.Uniform(low=1.0, high=20.0).sample([dim]),
        ),
        final_rotation="lu",
    )


def mtaf_rqs(dim: int, dfs: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF with a fixed base distribution consisting of normal distributions (for light-tailed marginals)
        and Student-t distributions with degrees of freedom given by dfs. """
    return flows.build_mtaf(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(fix_tails=True, tail_init=dfs['dfs']),
        final_rotation="lu",
    )


def comet(dim, dfs, model_config, x_trn):
    dfs = dfs['dfs']
    if isinstance(dfs, torch.Tensor):
        dfs = [float(df.cpu().numpy()) for df in dfs]

    return flows.build_comet(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            data=x_trn, 
            fix_tails=True, 
            tail_init=dfs
        ),
        final_rotation="lu",
    )


model_definitions = {
    "ttf": ttf_rqs,
    "ttf_fix": ttf_rqs_fix,
    "ttf_mod": ttf_rqs_mod,
    "gtaf": gtaf_rqs,
    "mtaf": mtaf_rqs,
    "normal":  normal,
    "normal_preprocess": normal,
    "comet": comet,
}

"""
Experiment code
"""

def run_experiment(
    data_source: str, # "insurance", "fama5" or "sp500"
    experiment_name: str,
    split: int, # data split
    seed: int, # RNG seed
    model_label: str, # a key string in model_definitions
    opt_params: dict, # params for optimizer
    model_config: dict, # hyperparams for model architecture
    experiment_ix=None, # Not used
):
    # general setup
    out_path = f"{data_source}/{experiment_name}"
    loss_path = f"{get_experiment_output_path()}/{out_path}/losses"

    torch.manual_seed(seed)

    # prepare data
    x = real_data_sources[data_source]()
    n = x.shape[0]
    print(f"Num data samps: {n}")
    dim = x.shape[1]

    tail_path = f'{get_data_path()}/splits/{data_source}/{split}'
    if not Path(f"{tail_path}.p").is_file():
        raise Exception(
            f"Split data not present at {tail_path}.p, either configure "
            "TAILNFLOWS_DATA_DIR, or run `python experiments/density_estimation_real_data/generate_splits.py`"
        )
    
    splits_and_tail = load_raw_data(tail_path)["experiment_data"][0]
    tail_and_scale = splits_and_tail["metadata"]
    
    trn_ix = splits_and_tail["split"]["trn"]
    val_ix = splits_and_tail["split"]["val"]
    tst_ix = splits_and_tail["split"]["tst"]

    mean = torch.tensor(tail_and_scale["mean"])
    scale = torch.tensor(tail_and_scale["std"])

    x_trn = (x[trn_ix] - mean) / scale
    x_val = (x[val_ix] - mean) / scale
    x_tst = (x[tst_ix] - mean) / scale

    # create model and train
    if model_label.endswith('preprocess'):
        preprocessor = get_preprocessor(tail_and_scale["dfs"])
    else:
        preprocessor = None
    
    model_fcn = model_definitions[model_label]
    if model_label == 'comet':
        model_fcn = partial(model_fcn, x_trn=x_trn)

    label = f'{data_source}-{split}-{model_label}'
    
    model = model_fcn(
        dim,
        tail_and_scale,
        model_config,
    ).to(DEFAULT_DTYPE)

    fit_data = data_fit.train(
        model,
        x_trn.to(DEFAULT_DTYPE),
        x_val.to(DEFAULT_DTYPE),
        x_tst.to(DEFAULT_DTYPE),
        **opt_params,
        label=label,
        preprocess_transformation=preprocessor,
    )

    tst_loss, val_loss, tst_ix, losses, vlosses, steps, hook_data = fit_data

    loss_ix = add_raw_data(
        loss_path,
        label,
        {
            "losses": losses.detach().cpu(),
            "vlosses": vlosses.detach().cpu(),
            "steps": steps.detach().cpu(),
            "tst_ix": tst_ix,
        },
        force_write=True,
    )

    add_experiment_output_data(
        out_path,
        label,
        {
            "model": model_label,
            "dim": dim, 
            "seed": seed,
            "split": split,
            "tst_ll": float(tst_loss),
            "val_ll": float(val_loss),
            "tst_ix": tst_ix,
            "loss_path": loss_path,
            "loss_ix": loss_ix,
            **opt_params,
            **model_config,
        },
        force_write=True,
    )


optimisation_overrides = {
    'sp500': {
        "lr": 5e-4, 
        "num_steps": 2_000, 
        "batch_size": 100, 
        "early_stop_patience": 500,
        "eval_period": 25,
        "lr_scheduler": None,
    },
    'fama5': {
        "lr": 5e-4, 
        "num_steps": 5_000, 
        "batch_size": 100, 
        "early_stop_patience": 500,
        "eval_period": 25,
        "lr_scheduler": None,
    }
}

def configured_experiments():
    """ Run several experiments in parallel by modifying the following dictionaries. """

    model_labels = [
        # "normal", 
        "ttf_fix", 
        "ttf_mod",
        # "ttf", 
        # "comet", 
        # "mtaf", 
        # "gtaf"
    ]

    experiment_name = "2026-08-26-de-sp500-try01"
    # data_sources = ['fama5', 'sp500', 'insurance']
    data_sources = ['sp500']

    opt_params = {
        "lr": 5e-4, 
        "num_steps": 3_000, 
        "batch_size": 512,
        "early_stop_patience": 1_00,
        "eval_period": 20, # period for computing validation loss
        "lr_scheduler": "cosine_anneal_wr",
    }

    model_config = {
        "depth": 1,
        "num_bins": 5,
        "tail_bound": 2.5
    }

    experiments = []
    for data_source in data_sources:
        for split in range(10):
            repeat_seed = 17*split
            for depth in [1]:
                for model_label in model_labels:
                    
                    opt_params = optimisation_overrides.get(data_source, opt_params)

                    experiments.append(dict(
                        data_source=data_source,
                        experiment_name=experiment_name,
                        split=split,
                        seed=repeat_seed,
                        model_label=model_label,
                        opt_params=opt_params,
                        model_config=model_config,
                    ))

    parallel_runner(run_experiment, experiments, max_runs=5)
    # print(experiments)
    # run_experiment(**experiments[0])

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    configured_experiments()
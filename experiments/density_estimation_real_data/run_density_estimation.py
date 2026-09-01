from pathlib import Path
from functools import partial

import torch

from nflows.transforms.base import Transform

from tailnflows.targets.data import real_data_sources
from tailnflows.models import flows
from tailnflows.train import data_fit
from tailnflows.utils import add_raw_data, get_experiment_output_path, load_raw_data, get_data_path, add_experiment_output_data, parallel_runner
from tailnflows.models.preprocessing import inverse_and_lad as t_to_norm_inverse_and_lad
from tailnflows.models.extreme_transformations import NNKwargs, SpecifiedNNKwargs, configure_nn

DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)

gpu_ix = 3
if torch.cuda.is_available():
    torch.set_default_device(f"cuda:{gpu_ix}")
    DEFAULT_DEVICE = torch.device(f"cuda:{gpu_ix}")
else:
    torch.set_default_device("cpu")
    DEFAULT_DEVICE = torch.device("cpu")

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
        u_linear_layer=True,
        nn_kwargs=NNKwargs(
            hidden_features=model_config.get("hidden_features"),
            num_blocks=model_config.get("num_blocks"),
            use_batch_norm=model_config.get("use_batch_norm"),
        ),
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


def ttf_rqs(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final TTF transformation with learnable tail params."""
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
        ),
        final_rotation="lu",
    )


def ttf_rqs_fix(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final TTF transformation with non-learnable tail params,
        initialized symmetrically for pos and neg tails given by the values in dfs. """
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=True,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
        ),
        final_rotation="lu",
    )


def ttf_rqs_hdonly(dim: int, metadata, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final TTF transformation (only in heavy-tailed directions) with learnable tail params, initialized by sampling uniformly from [0.05, 1.0].
        As the tail param are initialized randomly, the dfs argument is non-used. """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['neg_dfs']],
            hd_only=True,
            mod="std",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def ttf_rqs_lin(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final linear modified TTF transformation with learnable tail params given by dfs. """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            hd_only=False,
            mod="lin",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def ttf_rqs_lin_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final linear modified TTF transformation (only in heavy-tailed directions)
    with learnable tail params given by dfs. """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['neg_dfs']],
            hd_only=True,
            mod="lin",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def ttf_rqs_qua(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final quadratic-derivative-modified TTF transformation with fixed tail params given by dfs. """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=True,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            hd_only=False,
            mod="qua",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def ttf_rqs_qua_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final quadratic-derivative-modified TTF transformation (only in heavy-tailed directions)
    with fixed tail params given by dfs. """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=True,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['neg_dfs']],
            hd_only=True,
            mod="qua",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def ttf_rqs_erfi(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final erfi-modified TTF transformation with learnable tail params given by dfs. """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['dfs']],
            hd_only=False,
            mod="erfi",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def ttf_rqs_erfi_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final erfi-modified TTF transformation (in heavy-tailed directions only)
    with learnable tail params given by dfs. """
    return flows.build_ttf_mod_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.0 for df in metadata['neg_dfs']],
            hd_only=True,
            mod="erfi",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def gtaf_rqs(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
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


def mtaf_rqs(dim: int, metadata: dict, model_config: dict, device: torch.device = DEFAULT_DEVICE) -> flows.ExperimentFlow:
    """ Builds a NSF with a fixed base distribution consisting of normal distributions (for light-tailed marginals)
        and Student-t distributions with degrees of freedom given by dfs. """
    return flows.build_mtaf(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(fix_tails=True, tail_init=metadata['dfs']),
        final_rotation="lu",
        device=device,
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
    "ttf_hdonly": ttf_rqs_hdonly,
    "ttf_fix": ttf_rqs_fix,
    "ttf_lin": ttf_rqs_lin,
    "ttf_lin_hdonly": ttf_rqs_lin_hdonly,
    "ttf_qua": ttf_rqs_qua,
    "ttf_qua_hdonly": ttf_rqs_qua_hdonly,
    "ttf_erfi": ttf_rqs_erfi,
    "ttf_erfi_hdonly": ttf_rqs_erfi_hdonly,
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
    data_source: str, # "insurance", "fama5", "sp500" or "climate"
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

    ####################################
    # prepare train, val and test data #
    ####################################

    tail_path = f'{get_data_path()}/splits/{data_source}/{split}'
    if not Path(f"{tail_path}.p").is_file():
        raise Exception(
            f"Split data not present at {tail_path}.p, either configure "
            "TAILNFLOWS_DATA_DIR, or run `python experiments/density_estimation_real_data/generate_splits.py`"
        )

    splits_and_tail = load_raw_data(tail_path)["experiment_data"][0]
    metadata = splits_and_tail["metadata"]

    if data_source == "climate":
        # Load train/val/test split directly as normalized tensors
        x_trn = splits_and_tail["split"]["x_trn"]
        x_val = splits_and_tail["split"]["x_val"]
        x_tst = splits_and_tail["split"]["x_tst"]
        n = x_trn.shape[0] + x_val.shape[0] + x_tst.shape[0]
        print(f"Num data samps: {n}")
        dim = x_trn.shape[1]

    else:
        # Load whole data tensor and train/val/test indices. Then split and normalize.
        x = real_data_sources[data_source]()
        n = x.shape[0]
        print(f"Num data samps: {n}")
        dim = x.shape[1]
        
        trn_ix = splits_and_tail["split"]["trn"]
        val_ix = splits_and_tail["split"]["val"]
        tst_ix = splits_and_tail["split"]["tst"]

        mean = torch.tensor(metadata["mean"])
        scale = torch.tensor(metadata["std"])

        x_trn = (x[trn_ix] - mean) / scale
        x_val = (x[val_ix] - mean) / scale
        x_tst = (x[tst_ix] - mean) / scale

    ##########################
    # create model and train #
    ##########################

    # climate: Force heavy tails (with max df = 10.0) for the last 177 marginals (see mTAF paper)
    if data_source == "climate":
        metadata["dfs"] = [(lambda nu: 10.0 if (nu > 10.0 or nu == 0.0) else nu)(df) for df in metadata["dfs"]]
        metadata["pos_dfs"] = [(lambda nu: 10.0 if (nu > 10.0 or nu == 0.0) else nu)(df) for df in metadata["pos_dfs"]]
        metadata["neg_dfs"] = [(lambda nu: 10.0 if (nu > 10.0 or nu == 0.0) else nu)(df) for df in metadata["neg_dfs"]]

    # Load preprocess if specified
    if model_label.endswith('preprocess'):
        preprocessor = get_preprocessor(metadata["dfs"])
    else:
        preprocessor = None

    # Load the specified model
    model_fcn = model_definitions[model_label]
    if model_label == 'comet':
        model_fcn = partial(model_fcn, x_trn=x_trn) # comet flow model function needs training data as additional argument
    model = model_fcn(
        dim,
        metadata,
        model_config,
    ).to(DEFAULT_DTYPE)

    label = f'{data_source}-{split}-{model_label}'

    # Train and evaluate model
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

    ################
    # Save results #
    ################

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
            "tst_nll": float(tst_loss),
            "val_nll": float(val_loss),
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
        "batch_size": 512, 
        "early_stop_patience": 500,
        "eval_period": 25,
        "lr_scheduler": None,
    },
    'fama5': {
        "lr": 5e-4, 
        "num_steps": 8_000, 
        "batch_size": 512, 
        "early_stop_patience": 500,
        "eval_period": 25,
        "lr_scheduler": None,
    },
    'insurance': {
        "lr": 5e-4, 
        "num_steps": 2_000, 
        "batch_size": 512, 
        "early_stop_patience": 500,
        "eval_period": 25,
        "lr_scheduler": None,
    },
    'climate': {
        "lr": 1e-4, 
        "num_steps": 20_000, 
        "batch_size": 512,
        "early_stop_patience": 5_00,
        "eval_period": 25,
        "lr_scheduler": "cosine_anneal_wr",
    },
}

def configured_experiments():
    """ Run several experiments in parallel by modifying the following dictionaries. """

    model_labels = [
        # "normal", 
        # "ttf",
        # "ttf_hdonly",
        # "ttf_fix", 
        # "ttf_lin",
        # "ttf_lin_hdonly",
        # "ttf_qua",
        # "ttf_qua_hdonly",
        # "ttf_erfi",
        # "ttf_erfi_hdonly",
        "mtaf", 
        "gtaf"
    ]

    experiment_name = "2026-08-31-de-mtaf-test"
    # data_sources = ['climate', 'fama5', 'sp500', 'insurance']
    data_sources = ['insurance']

    opt_params = { # currently will be overriden by optimization_overrides
        "lr": 1e-4, 
        "num_steps": 20_000, 
        "batch_size": 512,
        "early_stop_patience": 5_00,
        "eval_period": 25, # period for computing validation loss
        "lr_scheduler": "cosine_anneal_wr",
    }

    # model_config
    # depths = [1, 2]
    depths = [1]
    numbers_of_bins = [5]
    tail_bounds = [2.5] # for RQS layers, not final tail trafos!

    experiments = []
    for data_source in data_sources:
        for split in range(10):
            repeat_seed = 17*split

            if data_source == "climate": # climate data uses different model config

                model_config = {
                    "depth": 5,
                    "num_bins": 3,
                    "tail_bound": 2.5,
                    "num_blocks": 2,
                    "hidden_features": 100,
                    "use_batch_norm": True,
                }

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

            else:
                for depth in depths:
                    for num_bins in numbers_of_bins:
                        for tail_bound in tail_bounds:

                            model_config = {
                                "depth": depth,
                                "num_bins": num_bins,
                                "tail_bound": tail_bound,
                            }

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

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    configured_experiments()
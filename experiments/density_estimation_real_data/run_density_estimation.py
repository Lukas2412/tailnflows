from pathlib import Path
from functools import partial
from typing import Optional

import torch

DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)

gpu_ix = 2
if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device(f"cuda:{gpu_ix}")
    torch.cuda.set_device(gpu_ix) # tensors with device="cuda" will be put on cuda{gpu_ix}
else:
    DEFAULT_DEVICE = torch.device("cpu")
    torch.set_default_device("cpu")

from nflows.transforms.base import Transform

from tailnflows.targets.data import real_data_sources
from tailnflows.models import flows
from tailnflows.train import data_fit
from tailnflows.utils import add_raw_data, get_experiment_output_path, load_raw_data, get_data_path, add_experiment_output_data, parallel_runner
from tailnflows.models.preprocessing import inverse_and_lad as t_to_norm_inverse_and_lad
from tailnflows.models.extreme_transformations import NNKwargs
from tailnflows.models.utils import invertibility_check
from tailnflows.metrics import metrics

"""
Model specifications
"""

def base_rqs_spec(dim: int, model_config: dict, num_light: Optional[int] = None, num_heavy: Optional[int] = None) -> list[Transform]:
    """ Builds a list of transformations representing an NSF with {depth} rational quadratic spline (rqs) layers and LU layers. """
    return flows.base_nsf_transform(
        dim, 
        num_bins=model_config.get('num_bins', 5),
        tail_bound=model_config.get('tail_bound', 3),
        affine_autoreg_layer=True,
        depth=model_config.get('depth', 1),
        u_linear_layer=True, # alternate rqs layers with LU layers (for depth >= 2. A final LU rotation is appended regardless of depth).
        nn_kwargs=NNKwargs(
            hidden_features=model_config.get("hidden_features"),
            num_blocks=model_config.get("num_blocks"),
            use_batch_norm=model_config.get("use_batch_norm"),
        ),
        mtaf=model_config.get('mtaf', False),
        num_light=num_light,
        num_heavy=num_heavy,
    )

def normal(dim: int, dfs, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model without a final tail transformation. """
    return flows.build_base_model(
        dim, 
        use="density_estimation", 
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        final_rotation="lu",
        device=DEFAULT_DEVICE,
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
        return z.to(dtype=DEFAULT_DTYPE, device=x.device), lad.to(DEFAULT_DTYPE).to(x.device)

    return _preprocess


def ttf_rqs(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final TTF transformation with learnable tail params."""
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['neg_dfs']],
        ),
        final_rotation="lu",
    )


def ttf_rqs_fix(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final TTF transformation with non-learnable tail params. """
    return flows.build_ttf_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=True,
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['neg_dfs']],
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
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['neg_dfs']],
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
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['neg_dfs']],
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
            pos_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 1e-4 for df in metadata['neg_dfs']],
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


def softlog_rqs(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final softlog transformation. """
    return flows.build_softlog_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['neg_dfs']],
            hd_only=False,
            mod="std",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def softlog_rqs_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final softlog transformation (only in heavy-tailed directions). """
    return flows.build_softlog_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
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


def softlog_rqs_lin(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final linear-modified softlog transformation. """
    return flows.build_softlog_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['neg_dfs']],
            hd_only=False,
            mod="lin",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def softlog_rqs_lin_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final linear-modified softlog transformation (only in heavy-tailed directions). """
    return flows.build_softlog_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
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


def arcsinh_rqs(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final arcsinh transformation. """
    return flows.build_arcsinh_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['neg_dfs']],
            hd_only=False,
            mod="std",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def arcsinh_rqs_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final arcsinh transformation (only in heavy-tailed directions). """
    return flows.build_arcsinh_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
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


def arcsinh_rqs_lin(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final linear-modified arcsinh transformation. """
    return flows.build_arcsinh_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['neg_dfs']],
            hd_only=False,
            mod="lin",
            a_pos_init=None,
            a_neg_init=None,
            fix_params=False,
        ),
        final_rotation="lu",
    )


def arcsinh_rqs_lin_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final linear-modified arcsinh transformation (only in heavy-tailed directions). """
    return flows.build_arcsinh_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
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


def softarcsinh_rqs(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final softarcsinh transformation. """
    return flows.build_softarcsinh_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['neg_dfs']],
            hd_only=False,
            b_pos_init=[1.0 for _ in range(dim)],
            b_neg_init=[1.0 for _ in range(dim)],
            fix_params=False,
        ),
        final_rotation="lu",
    )


def softarcsinh_rqs_hdonly(dim: int, metadata: dict, model_config: dict) -> flows.ExperimentFlow:
    """ Builds a NSF model with a final softarcsinh transformation. """
    return flows.build_softarcsinh_m(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            device=DEFAULT_DEVICE,
            pos_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['pos_dfs']],
            neg_tail_init=[1 / df if df != 0.0 else 0.25 for df in metadata['neg_dfs']],
            hd_only=True,
            b_pos_init=[1.0 for _ in range(dim)],
            b_neg_init=[1.0 for _ in range(dim)],
            fix_params=False,
        ),
        final_rotation="lu",
    )


def gtaf_rqs(dim: int, metadata: dict, model_config: dict, device: torch.device = DEFAULT_DEVICE) -> flows.ExperimentFlow:
    """ Builds a NSF with a trainable Student-t base distribution with marginal dfs sampled uniformly from [1.0, 20.0]. """
    return flows.build_gtaf(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config),
        model_kwargs=dict(
            fix_tails=False,
            # tail_init=torch.distributions.Uniform(low=1.0, high=20.0).sample([dim]),
            tail_init=[df if (df != 0.0 and df <= 10.0) else 10.0 for df in metadata['dfs']] # in df terms
        ),
        final_rotation="lu",
        device=device,
    )


def mtaf_rqs(dim: int, metadata: dict, model_config: dict, device: torch.device = DEFAULT_DEVICE) -> flows.ExperimentFlow:
    """ Builds a NSF with a fixed base distribution consisting of normal distributions (for light-tailed marginals)
        and Student-t distributions with degrees of freedom given by dfs. """
    tail_init = [df if df <= 10.0 else 0.0 for df in metadata['dfs']] # in df terms
    num_light = int(sum(df == 0.0 for df in tail_init))
    num_heavy = dim - num_light
    model_config_mtaf = model_config.copy()
    model_config_mtaf["mtaf"] = True
    return flows.build_mtaf(
        dim,
        use="density_estimation",
        base_transformation_init=partial(base_rqs_spec, model_config=model_config_mtaf, num_light=num_light, num_heavy=num_heavy),
        model_kwargs=dict(
            fix_tails=True,
            tail_init=tail_init,
        ),
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
    "softlog": softlog_rqs,
    "softlog_hdonly": softlog_rqs_hdonly,
    "softlog_lin": softlog_rqs_lin,
    "softlog_lin_hdonly": softlog_rqs_lin_hdonly,
    "arcsinh": arcsinh_rqs,
    "arcsinh_hdonly": arcsinh_rqs_hdonly,
    "arcsinh_lin": arcsinh_rqs_lin,
    "arcsinh_lin_hdonly": arcsinh_rqs_lin_hdonly,
    "softarcsinh": softarcsinh_rqs,
    "softarcsinh_hdonly": softarcsinh_rqs_hdonly,
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
    save_samples: bool = False,
    save_model: bool = False,
):
    # general setup
    out_path = f"{data_source}/{experiment_name}"
    loss_path = f"{get_experiment_output_path()}/{out_path}/losses"
    results_path = f"{out_path}/results"

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
        metadata["dfs"][-177:] = [(lambda nu: 10.0 if (nu > 10.0 or nu == 0.0) else nu)(df) for df in metadata["dfs"][-177:]]
        metadata["pos_dfs"][-177:] = [(lambda nu: 10.0 if (nu > 10.0 or nu == 0.0) else nu)(df) for df in metadata["pos_dfs"][-177:]]
        metadata["neg_dfs"][-177:] = [(lambda nu: 10.0 if (nu > 10.0 or nu == 0.0) else nu)(df) for df in metadata["neg_dfs"][-177:]]

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
    ).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

    label = f'{data_source}-{split}-{model_label}'

    # opt_params["num_steps"] = opt_params["num_epochs"] * x_trn.shape[0] // opt_params["batch_size"]
    # opt_params_copy = opt_params.copy()
    # del opt_params_copy["num_epochs"]

    # Train and evaluate model
    # fit_data = data_fit.train(
    #     model,
    #     x_trn.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE),
    #     x_val.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE),
    #     x_tst.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE),
    #     **opt_params_copy,
    #     label=label,
    #     preprocess_transformation=preprocessor,
    #     device=DEFAULT_DEVICE,
    # )
    # tst_loss, val_loss, tst_ix, losses, vlosses, steps, hook_data = fit_data
    fit_data = data_fit.train_epochs(
        model,
        x_trn.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE),
        x_val.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE),
        x_tst.to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE),
        **opt_params,
        label=label,
        best_model_path=f"{get_experiment_output_path()}/{out_path}/best_models/{label}.pth",
        device=DEFAULT_DEVICE,
    )
    tst_loss, val_loss, losses, vlosses = fit_data

    # Invertibility check
    samps = torch.randn(2048, dim, device=DEFAULT_DEVICE)
    is_invertible = invertibility_check(model, samps)
    print(f"Inveritbility check: {is_invertible}")

    # Compute metrics
    dfs_tensor = torch.tensor(metadata["dfs"])
    heavy_mask = dfs_tensor > torch.zeros_like(dfs_tensor)
    if data_source == "climate": # Climate: Classify last 177 marginals as HT
        heavy_mask[-177:] = True

    with torch.no_grad():
        num_samps = x_tst.shape[0]
        target_samps = x_tst.to(DEFAULT_DTYPE).detach().cpu()
        synth_samps = model.sample(num_samps).detach().cpu()

        print("Compute sliced Wasserstein-2 metric ...")
        sw2 = metrics.sliced_wp(synth_samps, target_samps, p=2.0)
        print("Compute Wasserstein-1 metrics ...")
        w1_ht = metrics.wp_1d_mean_over_dims(synth_samps, target_samps, dims=heavy_mask, p=1.0)
        w1_lt = metrics.wp_1d_mean_over_dims(synth_samps, target_samps, dims=~heavy_mask, p=1.0)
        print("Compute relative quantile errors ...")
        var_rel_err_ht = metrics.extreme_quantile_rel_error_multiq_over_dims(synth_samps, target_samps, heavy_mask)
        var_rel_err_lt = metrics.extreme_quantile_rel_error_multiq_over_dims(synth_samps, target_samps, ~heavy_mask)


    ################
    # Save results #
    ################

    print("Save results ...")

    loss_ix = add_raw_data(
        loss_path,
        label,
        {
            "losses": losses.detach().cpu(),
            "vlosses": vlosses.detach().cpu(),
            # "steps": steps.detach().cpu(),
            # "tst_ix": tst_ix,
        },
        force_write=True,
    )

    output_dict = {
        "model": model_label,
        "dim": dim, 
        "seed": seed,
        "split": split,
        "tst_nll": float(tst_loss),
        "val_nll": float(val_loss),
        "sw2": sw2, # Sliced Wasserstein-2
        "w1_ht": w1_ht, # Avg wasserstein-1 on HT dims
        "w1_lt": w1_lt, # Avg wasserstein-1 on LT dims
        "var99_ht": var_rel_err_ht[0].item(), # Avg rel VaR_99 difference on HT dims
        "var995_ht": var_rel_err_ht[1].item(),
        "var999_ht": var_rel_err_ht[2].item(),
        "var99_lt": var_rel_err_lt[0].item(), # Avg rel VaR_99 difference on HT dims
        "var995_lt": var_rel_err_lt[1].item(),
        "var999_lt": var_rel_err_lt[2].item(),
        # "tst_ix": tst_ix,
        # "loss_path": loss_path,
        # "loss_ix": loss_ix,
        "model_str": model.__repr__(),
        **opt_params,
        **model_config,
    }

    output_dict["final_trafo_type"] = type(model.get_final_transformation())
    output_dict["final_trafo_params"] = list(model.get_final_transformation().named_parameters())

    if save_samples:
        output_dict["synth_samps"] = synth_samps

    if save_model:
        output_dict["model"] = model

    add_experiment_output_data(
        results_path ,
        label,
        output_dict,
        force_write=True,
    )

    # NOTE: Try this to avoid OOM
    if DEFAULT_DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("Experiment completed.")


optimisation_overrides = {
    'sp500': {
        "lr": 5e-4, 
        "num_epochs": 400,
        "batch_size": 512, 
        "early_stop_patience": 500,
        "eval_period": 1,
        "lr_scheduler": None,
    },
    'fama5': {
        "lr": 5e-4, 
        "num_epochs": 400, 
        "batch_size": 512, 
        "early_stop_patience": 500,
        "eval_period": 1,
        "lr_scheduler": None,
    },
    'insurance': {
        "lr": 5e-4, 
        "num_epochs": 400,
        "batch_size": 512, 
        "early_stop_patience": 500,
        "eval_period": 1,
        "lr_scheduler": None,
    },
    'climate': {
        "lr": 1e-4, 
        "num_steps": 20_000,
        "batch_size": 512,
        "early_stop_patience": 5_00,
        "eval_period": 1,
        "lr_scheduler": "cosine_anneal_wr",
    },
}

def configured_experiments():
    """ Run several experiments in parallel by modifying the following code. """

    model_labels = [
        "normal",
        # # "ttf",
        # "ttf_hdonly",
        # # # "ttf_fix", 
        # # # "ttf_lin",
        # "ttf_lin_hdonly",
        # # # "ttf_qua",
        # "ttf_qua_hdonly",
        # # # "ttf_erfi",
        # "ttf_erfi_hdonly",
        # # # "softlog",
        # "softlog_hdonly",
        # # # "softlog_lin",
        # "softlog_lin_hdonly",
        # # # "arcsinh",
        # "arcsinh_hdonly",
        # # # "arcsinh_lin",
        # "arcsinh_lin_hdonly",
        # "softarcsinh",
        # "softarcsinh_hdonly",
        # "mtaf", 
        # "gtaf",
        # "comet", # comet currently does not work
    ]

    experiment_name = "2026-09-17-train-test"
    # data_sources = ['climate', 'fama5', 'sp500', 'insurance']
    data_sources = ['insurance']

    opt_params = {
        "lr": 1e-4,
        "num_epochs": 5000,
        "batch_size": 512,
        "early_stop_patience": 500,
        # "eval_period": 1, # period for computing validation loss
        "lr_scheduler": None,
        "grad_clip": 10.0,
    }

    # model_config
    depths = [1] # flow architecture from TTF paper
    numbers_of_bins = [5]
    tail_bounds = [2.5] # for RQS layers, not final tail trafos!

    experiments = []
    print("Setting up experiment plan...")
    for data_source in data_sources:
        for split in range(1): # NOTE -> 10
            repeat_seed = 53*split + 1000

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

                    # NOTE: Uncomment to use default (TTF paper) settings
                    # opt_params = optimisation_overrides.get(data_source, opt_params)

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
                                "use_batch_norm": True, # for stability of NSFs
                            } # hidden dimension will be feature_dim + 10

                            for model_label in model_labels:

                                # NOTE: Uncomment to use default (TTF paper) settings
                                # opt_params = optimisation_overrides.get(data_source, opt_params)

                                experiments.append(dict(
                                    data_source=data_source,
                                    experiment_name=experiment_name,
                                    split=split,
                                    seed=repeat_seed,
                                    model_label=model_label,
                                    opt_params=opt_params,
                                    model_config=model_config,
                                ))

    parallel_runner(partial(run_experiment, save_samples=True), experiments, max_runs=5)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    configured_experiments()
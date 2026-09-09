import torch
import numpy as np
import ot
from tailnflows.metrics.psis import gpdfitnew


def bootstrap_metric(log_p_x, log_q_x, replications, metric):
    samples = len(log_p_x)
    replicate_ix = torch.randint(samples, (replications, samples))
    mean = metric(log_p_x, log_q_x)
    replications = torch.vmap(metric)(log_p_x[replicate_ix], log_q_x[replicate_ix])
    lower_rep, upper_rep = torch.quantile(replications, torch.tensor([0.05, 0.95]))
    lower = 2 * mean - upper_rep
    upper = 2 * mean - lower_rep
    ci = (lower, mean, upper)
    return tuple(float(v.detach()) for v in ci)


def ess(log_p_x, log_q_x):
    """
    Produces an ESS based sample efficiency metric.
    Usually between 0 and 1, anything not approaching 0 could be
    workable, depending on the situation.
    """
    log_w = log_p_x - log_q_x
    samples = len(log_w)
    log_norm = torch.logsumexp(log_w, 0)
    log_norm_iw = log_w - log_norm
    ess_efficiency = 1 / torch.exp(2 * log_norm_iw).sum()
    ess_efficiency = ess_efficiency / samples
    return ess_efficiency


def elbo(log_p_x, log_q_x):
    log_w = log_p_x - log_q_x
    return log_w.mean()


def entropy(log_p_x, log_q_x):
    return log_q_x.mean()


def marginal_likelihood(log_p_x, log_q_x):
    log_w = log_p_x - log_q_x
    samples = len(log_w)
    return torch.logsumexp(log_w - torch.log(torch.tensor([samples])), dim=0)


def psis_index(log_p_x, log_q_x):
    """
    Produces an PSIS (pareto smoothed importance sampling) score.
    This is the tail index of density ratios q/p, lower is better,
    below 0.7 is considered good enough for importance sampling.
    """
    log_w = log_p_x - log_q_x
    samples = len(log_w)
    M = int(min(3 * np.sqrt(samples), samples / 5))

    max_log_iw = log_w.max()
    log_w -= max_log_iw
    sorted_log_iw = torch.sort(log_w).values  # ascending
    tail_log_iw = sorted_log_iw[-M:]
    threshold = sorted_log_iw[-M - 1]

    tail_iw_exceedences = torch.exp(threshold) * torch.expm1(tail_log_iw - threshold)
    k, _ = gpdfitnew(tail_iw_exceedences.detach().cpu().numpy())
    return k


def avrg_nll(model, test_samps: torch.Tensor, features: int) -> float:
    """ Compute average NLL per sample and dimension.

    Args:
        model: Must implement a log_prob() method.
        test_samps (torch.Tensor): Batch of test samples on which to compute likelihood (Shape [batch, features]).
        features (int): number of features.

    Returns:
        nll (float): Average NLL per sample and dimension.
    """
    assert test_samps.shape[-1] == features, "Invalid shape"
    log_prob = model.log_prob(test_samps)
    if log_prob.shape == test_samps.shape: # log_prob wasn't summed over dimensions: So do that here
        log_prob = log_prob.sum(dim=-1)
    num_samps = len(log_prob)
    nll = - float(log_prob.sum().detach()) / (num_samps * features)
    return nll


def sliced_wp(synth_samps: torch.Tensor, target_samps: torch.Tensor, p: float = 1.0) -> float:
    """ Estimate the Wasserstein-p distance between synthetic distribution and target distribution
        using random 1d projections on synthetic and true samples.

    Args:
        synth_samps (torch.Tensor): Batch of synthetic samples (shape: [batch, features]).
        target_samps (torch.Tensor): Batch of target samples (shape: [batch, features]).
        p (float): Power for computing Wasserstein distance Wp. Defaults to 1.

    Returns:
        swp (float): Sliced Wp cost between synth and target samps.
    """
    swp = ot.sliced.sliced_wasserstein_distance(
        synth_samps.detach().cpu().numpy(),
        target_samps.detach().cpu().numpy(),
        n_projections=100,
        p=p
    )
    return float(swp)


def wp_1d_mean_over_dims(
    synth_samps: torch.Tensor,
    target_samps: torch.Tensor,
    dims: torch.Tensor,
    p: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Mean 1D empirical Wasserstein-p distance over selected marginals.

    Assumes uniform empirical weights and equal sample sizes.

    Computes:
        mean_{j in dims} W_p(synth[:, j], target[:, j])

    Args:
        synth_samps: Tensor of shape [n, d].
        target_samps: Tensor of shape [n, d].
        dims: Boolean mask of shape [d].
        p: Wasserstein order, usually >= 1.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor on same device as inputs.
    """
    assert synth_samps.ndim == 2
    assert target_samps.ndim == 2
    assert synth_samps.shape == target_samps.shape
    assert dims.shape[0] == synth_samps.shape[1]
    assert p >= 1

    dims = dims.to(device=synth_samps.device, dtype=torch.bool)

    if dims.sum() == 0:
        return 0.0

    # Select dimensions: [n, d_selected]
    x = synth_samps[:, dims]
    y = target_samps[:, dims]

    # Sort each selected marginal independently along sample axis.
    x_sorted, _ = torch.sort(x, dim=0)
    y_sorted, _ = torch.sort(y, dim=0)

    # Per-dimension W_p^p:
    # [d_selected]
    wp_p_per_dim = torch.mean(torch.abs(x_sorted - y_sorted).pow(p), dim=0)

    # Per-dimension W_p:
    # [d_selected]
    wp_per_dim = torch.clamp(wp_p_per_dim, min=eps).pow(1.0 / p)

    # Average over selected dimensions.
    return float(wp_per_dim.mean().detach())


def extreme_quantile_rel_error_multiq_over_dims(
    synth_samps: torch.Tensor,
    target_samps: torch.Tensor,
    dims: torch.Tensor,
    qs=(0.99, 0.995, 0.999),
    eps: float = 1e-12,
    reduction: str = "mean_dims",
) -> torch.Tensor:
    """Extreme quantile relative errors for multiple quantile levels.

    Computes:

        rel_err[k, j]
        =
        | Q_{qs[k]}(synth[:, j]) - Q_{qs[k]}(target[:, j]) |
        / (|Q_{qs[k]}(target[:, j])| + eps)

    Args:
        synth_samps:
            Synthetic samples, shape [n_synth, features].
        target_samps:
            Target samples, shape [n_target, features].
        dims:
            Boolean mask selecting dimensions, shape [features].
        qs:
            Iterable of quantile levels.
        eps:
            Small constant for numerical stability.
        reduction:
            "none":
                Return tensor of shape [num_qs, num_selected_dims].
            "mean_dims":
                Average over dimensions; return shape [num_qs].
            "mean_all":
                Average over quantiles and dimensions; return scalar.
            "sum_dims":
                Sum over dimensions; return shape [num_qs].

    Returns:
        Tensor depending on reduction.
    """
    assert synth_samps.ndim == 2
    assert target_samps.ndim == 2
    assert synth_samps.shape[1] == target_samps.shape[1]
    assert dims.shape[0] == synth_samps.shape[1]

    device = synth_samps.device
    dtype = synth_samps.dtype

    dims = dims.to(device=device, dtype=torch.bool)
    qs_t = torch.as_tensor(qs, device=device, dtype=dtype)

    assert torch.all((qs_t >= 0.0) & (qs_t <= 1.0))

    if dims.sum() == 0:
        if reduction == "none":
            return synth_samps.new_empty((len(qs), 0))
        elif reduction in {"mean_dims", "sum_dims"}:
            return synth_samps.new_zeros((len(qs),))
        elif reduction == "mean_all":
            return synth_samps.new_tensor(0.0)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    x = synth_samps[:, dims]
    y = target_samps[:, dims]

    # Shapes: [num_qs, num_selected_dims]
    q_synth = torch.quantile(x, qs_t, dim=0)
    q_target = torch.quantile(y, qs_t, dim=0)

    rel_err = torch.abs(q_synth - q_target) / (torch.abs(q_target) + eps)

    if reduction == "none":
        return rel_err
    elif reduction == "mean_dims":
        return rel_err.mean(dim=1)
    elif reduction == "sum_dims":
        return rel_err.sum(dim=1)
    elif reduction == "mean_all":
        return rel_err.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
import numpy as np
import torch
from tqdm import tqdm


def compute_tvar_np(data: np.ndarray, alpha: float = 0.95) -> float:
    """
    Compute the tail value at risk (tvar) for absolute values of the data (considering both tails).

    Args:
        data (np.ndarray): array containing the data
        alpha (float): quantile level for tvar computation

    Returns:
        tvar (float): tvar value
    """
    sorted_data = np.sort(np.abs(data))
    tvar = np.mean(sorted_data[int(alpha*len(sorted_data)):])
    return float(tvar)


def compute_tvar_torch(data: torch.Tensor, alpha: float = 0.95) -> float:
    """
    Compute the tail value at risk (tvar) for absolute values of the data (considering both tails).

    Args:
        data (torch.Tensor): tensor containing the data
        alpha (float): quantile level for tvar computation

    Returns:
        tvar (float): tvar value
    """
    sorted_data, _ = torch.sort(torch.abs(data.squeeze()))
    tvar = torch.mean(sorted_data[int(alpha*len(sorted_data)):])
    return tvar.item()


def compute_arealoglog_np(data_true: np.ndarray, data_synth: np.ndarray) -> float:
    """
    Computes the area under the log-log plot.

    Args:
        data_true (np.ndarray): array containing the true data
        data_synth (np.ndarray): array containing synthetic data

    Returns:
        area (float): area under the log-log plot
    """
    n = len(data_true)
    area = 0.0
    eps = 1e-10  # minimum value to prevent log(0)
    for j in tqdm(range(n)):
        i = j + 1
        q_true = max(np.quantile(np.abs(data_true), 1 - i/n), eps)
        q_synth = max(np.quantile(np.abs(data_synth), 1 - i/n), eps)
        area += np.abs(np.log(q_true) - np.log(q_synth)) * np.log((i + 1)/i)
    return area


def compute_arealoglog_torch(data_true: torch.Tensor, data_synth: torch.Tensor, device: torch.device) -> float:
    """
    Computes the area under the log-log plot.

    Args:
        data_true (torch.Tensor): tensor containing the true data
        data_synth (torch.Tensor): tensor containing synthetic data
        device (torch.device): device on which to perform computations

    Returns:
        area (float): area under the log-log plot
    """
    data_true = data_true.to(device)
    data_synth = data_synth.to(device)

    n = data_true.shape[0]
    area = torch.tensor(0.0, device=device)
    eps = 1e-10  # minimum value to prevent log(0)
    for j in tqdm(range(n)):
        i = j + 1
        q_true = torch.clamp(torch.quantile(torch.abs(data_true.squeeze()), 1 - i/n), min=eps)
        q_synth = torch.clamp(torch.quantile(torch.abs(data_synth.squeeze()), 1 - i/n), min=eps)
        area += torch.abs(torch.log(q_true) - torch.log(q_synth)) * torch.log(torch.tensor([(i + 1)/i], device=device))
    return area.item()


def compute_arealoglog_torch_vectorized(data_true: torch.Tensor, data_synth: torch.Tensor, device: torch.device) -> float:
    """
    Computes the area under the log-log plot in a vectorized manner (faster on GPU).

    Args:
        data_true (torch.Tensor): tensor containing the true data
        data_synth (torch.Tensor): tensor containing synthetic data
        device (torch.device): device on which to perform computations

    Returns:
        area (float): area under the log-log plot
    """
    data_true = data_true.to(device)
    data_synth = data_synth.to(device)

    n = data_true.shape[0]
    eps = 1e-10  # minimum value to prevent log(0)
    # Create a tensor of indices from 1 to n
    indices = torch.arange(1, n + 1, device=device)
    # Create a tensor of fractions (i + 1)/i
    fractions = (indices + 1) / indices
    # Compute the quantiles for all indices at once
    quantiles_true = torch.quantile(torch.abs(data_true.squeeze()), 1 - indices / n)
    quantiles_synth = torch.quantile(torch.abs(data_synth.squeeze()), 1 - indices / n)
    # Clamp quantiles to prevent log(0) which would give -inf
    quantiles_true = torch.clamp(quantiles_true, min=eps)
    quantiles_synth = torch.clamp(quantiles_synth, min=eps)
    # Compute the log of the quantiles
    log_quantiles_true = torch.log(quantiles_true)
    log_quantiles_synth = torch.log(quantiles_synth)
    # Compute the absolute difference of the logs
    log_diff = torch.abs(log_quantiles_true - log_quantiles_synth)
    # Compute the log of the fractions
    log_fractions = torch.log(fractions)
    # Compute the area by summing the element-wise product of log_diff and log_fractions
    area = torch.sum(log_diff * log_fractions)
    return area.item()


def tail_index_diff(lambda_true: float, lambda_synth: float) -> float:
    """
    Compute the absolute difference in tail indices.

    Args:
        lambda_true (float): (estimated) tail index of the true data
        lambda_synth (float): (estimated) tail index of the synthetic data

    Returns:
        diff (float): absolute difference in tail indices. Returns 0 if both distributions are light-tailed (lambda = 0.0) and np.inf if one distribution is light-tailed and the other is heavy-tailed.
    """

    if min(lambda_synth, lambda_true) <= 0.0 and max(lambda_true, lambda_true) > 0.0:
        return np.inf
    else:
        return abs(lambda_synth - lambda_true)
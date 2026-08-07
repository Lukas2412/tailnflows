import torch
from torch.nn.functional import softplus, relu, sigmoid
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module
from nflows.transforms import Transform
from tailnflows.models.utils import inv_sftplus, inv_sigmoid
from typing import TypedDict, Optional, Callable
import math
import numpy as np
import scipy
from tailnflows.models.simple_spline import (
    univariate_forward_rqs,
    univariate_inverse_rqs,
)

from nflows.transforms.splines import rational_quadratic
from nflows.transforms.splines.rational_quadratic import (
    forward_rational_quadratic_spline,
    inverse_rational_quadratic_spline,
    unconstrained_rational_quadratic_spline_forward,
    unconstrained_rational_quadratic_spline_inverse,
)

from tailnflows.models.simple_spline import forward_rqs, inverse_rqs

MAX_TAIL = 5.0
LOW_TAIL_INIT = 0.1
HIGH_TAIL_INIT = 0.9
SQRT_2 = math.sqrt(2.0)
PI = math.pi
SQRT_PI = math.sqrt(PI)
LOG_SQRT_PI = math.log(SQRT_PI)
MIN_ERFC_INV = 1e-6


class NNKwargs(TypedDict, total=False):
    hidden_features: int
    num_blocks: int
    use_residual_blocks: bool
    random_mask: bool
    activation: Callable
    dropout_probability: float
    use_batch_norm: bool


class SpecifiedNNKwargs(TypedDict, total=True):
    hidden_features: int
    num_blocks: int
    use_residual_blocks: bool
    random_mask: bool
    activation: Callable
    dropout_probability: float
    use_batch_norm: bool


def configure_nn(nn_kwargs: NNKwargs) -> SpecifiedNNKwargs:
    return {
        "hidden_features": nn_kwargs.get("hidden_features", 5),
        "num_blocks": nn_kwargs.get("num_blocks", 2),
        "use_residual_blocks": nn_kwargs.get("use_residual_blocks", True),
        "random_mask": nn_kwargs.get("random_mask", False),
        "activation": nn_kwargs.get("activation", relu),
        "dropout_probability": nn_kwargs.get("dropout_probability", 0.0),
        "use_batch_norm": nn_kwargs.get("use_batch_norm", False),
    }


class ExtremeActivation(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_dim = dim
        self._unc_mix = torch.nn.Parameter(torch.ones([dim, 3]))
        # with torch.no_grad():
        #     self._unc_mix[:, 0] = 5.0  # init at identity
        self._unc_params = torch.nn.Parameter(torch.ones([dim, 2]))
        self.mix = torch.nn.Softmax(dim=1)

    def params(self):
        params = torch.nn.functional.sigmoid(self._unc_params) * 2
        heavy_tail = params[..., 0]
        light_tail = params[..., 1]
        return heavy_tail, light_tail

    def forward(self, z):
        heavy_tail, light_tail = self.params()
        mix = self.mix(self._unc_mix)  # dim x 3

        z_heavy = (
            _extreme_transform_and_lad(z.abs(), heavy_tail)[0] * z.sign()
        )  # batch x dim
        z_light = (
            _extreme_inverse_and_lad(z.abs(), light_tail)[0] * z.sign()
        )  # batch x dim

        combo = mix[:, 0] * z + mix[:, 1] * z_heavy + mix[:, 2] * z_light

        return combo


class ExtremeNetwork(torch.nn.Module):
    def __init__(
        self, features, hidden_features, num_blocks, output_multiplier, **kwargs
    ):
        super().__init__()
        self.base_model = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            **kwargs,
        )
        self.extreme_activation = ExtremeActivation(features * output_multiplier)

    def forward(self, x, context=None):
        param_data = self.base_model(x, context)
        adjusted_param_data = self.extreme_activation(param_data)
        return adjusted_param_data


def _erfcinv(x):
    with torch.no_grad():
        x = torch.clamp(x, min=MIN_ERFC_INV)
    return -torch.special.ndtri(0.5 * x) / x.new_tensor(SQRT_2)


def _small_erfcinv(log_g):
    """
    Use series expansion for erfcinv(x) as x->0
    """
    log_z_sq = 2 * log_g

    inner = torch.log(log_g.new_tensor(2 / PI)) - log_z_sq
    inner -= (torch.log(log_g.new_tensor(2 / PI)) - log_z_sq).log()

    z = inner.sqrt() / log_g.new_tensor(SQRT_2)

    return z


def _stable_erfcinv(x, log_x):
    with torch.no_grad():
        standard_x = torch.clamp(x, min=MIN_ERFC_INV, max=None)
        small_log_x = torch.clamp(log_x, min=None, max=torch.tensor(MIN_ERFC_INV, device=log_x.device).log())

    z = torch.where(
        x > MIN_ERFC_INV,
        -torch.special.ndtri(0.5 * standard_x) / x.new_tensor(SQRT_2),
        _small_erfcinv(small_log_x),
    )

    # Do one Newton step for better accuracy near the breaking point
    denom = -2.0 / x.new_tensor(SQRT_PI) * torch.exp(-z.square())
    denom = torch.where(denom == 0.0, torch.finfo(z.dtype).tiny, denom) # avoid devision by 0
    z = z - (torch.special.erfc(z) - x) / denom

    return z


def _erfi_scipy(z: torch.Tensor) -> torch.Tensor:
    """
    Compute erfi(z) using scipy.special.erfi on CPU tensors.
    """
    z_np = z.detach().cpu().numpy()
    erfi_np = scipy.special.erfi(z_np)
    return torch.from_numpy(erfi_np).to(z.device)


def _erfi_complex(x):
    # erfi(x) = -i * erf(i x)
    # Use complex path; result is real for real x (imag part ~ 0)
    cdtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    ix = (1j * x.to(cdtype))
    val = torch.special.erf(ix)
    erfi_val = (-1j) * val
    return erfi_val.real.to(x.dtype)


def _erfi_inv_initial_guess(x: torch.Tensor) -> torch.Tensor:
    """
    Provide a stable initial guess for computing erfi^{-1}(x) using Halley's method.
    """
    s = torch.sign(x)
    x_abs = x.abs()

    guess = torch.empty_like(x)

    # Use series inversion (order 1) for small |x| < 1.5
    small = x_abs < 1.5
    if small.any():
        # Leading term: erfi(y) ≈ 2y/√π
        guess[small] = SQRT_PI / 2 * x[small]

    # Use asymptotic inversion for large |x| >= 1.5
    large = ~small
    if large.any():
        # Asymptotic expansion (first order) for large y: erfi(y) ~ e^{y^2} / (sqrt(pi) y)
        # Solve: x_abs ≈ e^{y^2} / (sqrt(pi) y)
        # Take log on both sides and approximate y^2 - log(y) with y^2 for large y:

        guess_large = torch.sqrt(torch.log(x_abs[large]) + LOG_SQRT_PI)
        guess[large] = s[large] * guess_large

    return guess


def _erfi_inv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """
    Compute erfi^{-1}(x) using:
    - hybrid initial guess (series for small, asymptotic for large)
    - Halley's method for refinement
    """

    y = _erfi_inv_initial_guess(x).detach().clone()

    for _ in range(iters):
        f = _erfi_scipy(y) - x                     # f(y)
        fp = 2.0 / SQRT_PI * torch.exp(y * y)      # f'(y)
        fpp = 2.0 * y * fp                         # f''(y)

        # Halley update:
        # y_{n+1} = y - (2 f f') / (2 (f')^2 - f f'')
        nom = 2 * f * fp
        denom = 2 * fp * fp - f * fpp
        y = y - nom / denom

    return y


def _erfi_inv_newton(y, max_iters=8):
    # TODO: Test if this works and is better or worse than the Halley implementation!
    # NOTE: I don't think this is correct!
    # Solve u such that erfi(u) = y via Newton iterations.
    # Monotone, strictly increasing, so Newton with a decent init converges fast.
    # Do math in float64 if input is float32 for stability.
    orig_dtype = y.dtype
    u = y.to(torch.float64)
    y64 = u.clone()

    # Initial guess:
    # - small |y|: linear approx erfi(u) ~ 2/√π u => u ~ y * √π/2
    # - large |y|: grow ~ sqrt(log(1 + c y^2))
    small = (y64.abs() <= 1.0)
    u0_small = y64 * math.sqrt(math.pi) / 2.0
    u0_large = y64.sign() * torch.sqrt(torch.clamp(torch.log1p((math.pi / 4.0) * y64 * y64), min=0.0))
    u = torch.where(small, u0_small, u0_large)

    # Newton iterations with clamped step
    # f(u) = erfi(u) - y, f'(u) = 2/√π * exp(u^2)
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    for _ in range(max_iters):
        f = _erfi_complex(u) - y64
        deriv = two_over_sqrt_pi * torch.exp(u * u)
        delta = f / deriv
        # clamp update to avoid wild jumps
        delta = torch.clamp(delta, min=-1.0, max=1.0)
        u = u - delta

    return u.to(orig_dtype)


###############################################################
# ===== auxiliary functions for modified TTF transforms ===== #
###############################################################

def _cbrt(x): # real-valued cubic root, sign-correct
    return torch.sign(x) * torch.pow(torch.abs(x), x.new_tensor(1.0 / 3.0))


def _const_like(x, val):
    return x.new_tensor(val)


def _compute_rt(a: torch.Tensor, tail_param: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes parameters r and t for erfi modified TTF transformation (on pos side).

    Args:
        a (torch.Tensor): positive parameter defining the range (on pos side) for erfi transformations for each marginal (shape: [features])
        tail_param (torch.Tensor): tail parameters (on pos side) for each marginal (shape: [features])
        scale (torch.Tensor): scale parameters for each marginal (shape: [features])

    Returns:
        (tuple): containing:
            r (torch.Tensor): r parameters (on pos side) for each marginal (shape: [features])
            t (torch.Tensor): t parameters (on pos side) for each marginal (shape: [features])
    """
    b = scale * SQRT_2/SQRT_PI * torch.exp(-a*a/2) * torch.pow(torch.erfc(torch.abs(a) / SQRT_2), -(tail_param+1))
    c = b * (-a + (tail_param+1) * SQRT_2/SQRT_PI * torch.exp(-a*a/2) * torch.pow(torch.erfc(a/SQRT_2), -1))
    t = c / (2*a*b)
    r = b * torch.exp(-c*a/(2*b))

    return r, t


def compute_c2_c0(a: torch.Tensor, tail_param: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes parameters c2 and c0 for quadratic-slope modified TTF transformation.

    Args:
        a (torch.Tensor): positive parameter defining the range (on pos side) for quadratic-slope transformations for each marginal (shape: [features])
        tail_param (torch.Tensor): tail parameters (on pos side) for each marginal (shape: [features])
        scale (torch.Tensor): scale parameters for each marginal (shape: [features])

    Returns:
        (tuple): containing:
            c2 (torch.Tensor): c2 parameters (on pos side) for each marginal (shape: [features])
            c0 (torch.Tensor): c0 parameters (on pos side) for each marginal (shape: [features])
    """
    b = scale * SQRT_2/SQRT_PI * torch.exp(-a*a/2) * torch.pow(torch.erfc(torch.abs(a) / SQRT_2), -(tail_param+1))
    c = b * (-a + (tail_param+1) * SQRT_2/SQRT_PI * torch.exp(-a*a/2) * torch.pow(torch.erfc(a/SQRT_2), -1))
    c2 = c / (2*a)
    c0 = b - c2 * a * a

    return c2, c0


def r_prime_np(x: np.ndarray, tail_param: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Derivative of the (scalar) TTF transformation at x for given tail_param and scale (numpy version).

    Args:
        x (np.ndarray): input value
        tail_param (np.ndarray): tail parameter
        scale (np.ndarray): scale parameter

    Returns:
        np.ndarray: derivative value at x
    """
    return scale * np.sqrt(2/np.pi) * np.exp(-x**2/2) * (scipy.special.erfc(np.abs(x) / np.sqrt(2)))**(-tail_param - 1)


def r_pp_np(x: np.ndarray, tail_param: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Second derivative of the (scalar) TTF transformation at x for given tail_param and scale (numpy version).

    Args:
        x (np.ndarray): input value
        tail_param (np.ndarray): tail parameter
        scale (np.ndarray): scale parameter

    Returns:
        np.ndarray: second derivative value at x
    """
    return r_prime_np(x, tail_param, scale) * ( -x + (tail_param + 1) * np.sign(x) * np.sqrt(2/np.pi) * np.exp(-x**2/2) * (scipy.special.erfc(np.abs(x) / np.sqrt(2)))**(-1) )


def compute_a(tail_param: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Computes 'a' parameters (on pos side) for each marginal for quadratic-slope TTF transformation from tail parameters using a numerical solver.

    Args:
        tail_param (np.ndarray): tail parameters (on pos side) for each marginal (shape: [features])
        scale (np.ndarray): scale parameters for each marginal (shape: [features])

    Returns:
        a_values (np.ndarray): computed 'a' parameters (on pos side) for each marginal (shape: [features])
    """
    a_min = 0.1 # minimum value for 'a' if numerical solver fails, should be decent with tail_param in (0,5)
    
    a_values = np.zeros_like(tail_param)
    for i in range(len(tail_param)):
        if not np.isnan(tail_param[i]): # computation only needed for heavy-tailed marginals
            # Use root_scalar to find the root of K(x) = 0
            sol = scipy.optimize.root_scalar(
                lambda x: np.sign(x) * (2 * (r_prime_np(x, tail_param[i], scale[i]) - 1) / r_pp_np(x, tail_param[i], scale[i]) - x),
                method = 'secant',
                x0 = 1.0,
                x1 = 5.0,
            )
            if sol.converged and sol.root > 0:
                a_values[i] = sol.root
            else:
                print(f"WARNING: Numerical solver for 'a' did not converge for tail_param={tail_param[i]}, scale={scale[i]}. Setting 'a' to minimum value {a_min}.")
                a_values[i] = a_min
    
    return a_values





def _shift_power_transform_and_lad(z, tail_param):
    transformed = (SQRT_2 / SQRT_PI) * (torch.pow(1 + z / tail_param, tail_param) - 1)
    lad = (tail_param - 1) * torch.log(1 + z / tail_param)
    lad += torch.log(z.new_tensor(SQRT_2 / SQRT_PI))
    return transformed, lad


def _shift_power_inverse_and_lad(x, tail_param):
    transformed = (
        (SQRT_PI / SQRT_2) * tail_param * (torch.pow(1 + x, 1 / tail_param) - 1)
    )
    lad = ((1 / tail_param) - 1) * torch.log(1 + x)
    lad -= torch.log(x.new_tensor(SQRT_2 / SQRT_PI))
    return transformed, lad


def _extreme_transform_and_lad(z, tail_param):
    g = torch.special.erfc(z / SQRT_2)
    log_g = torch.log(g)
    x = torch.expm1(-tail_param * log_g) / tail_param

    lad = torch.log(g) * (-tail_param - 1)
    lad -= 0.5 * torch.square(z)
    lad += torch.log(z.new_tensor(SQRT_2 / SQRT_PI))

    return x, lad


def _ttf_sym_value_and_logprime(z, lam):
    """
    TODO: Plot this against the original implementation for sanity!
    R_lambda(z) and log|R_lambda'(z)| for symmetric-λ TTF with mu=0, sigma=1.
    Equations:
    g = erfc(|z|/√2)
    R(z) = sign(z) * (g^(-lam)-1)/lam
    R'(z) = g^(-lam-1) * (√2/√π) * exp(-z^2/2) (even function)
    Uses stable expm1/log forms.
    """
    u = torch.abs(z)
    g = torch.special.erfc(u / _const_like(u, math.sqrt(2.0)))
    log_g = torch.log(g)
    # Stable value
    val_abs = torch.expm1(-lam * log_g) / lam # (g^(-lam)-1)/lam
    val = torch.sign(z) * val_abs
    # Log-derivative
    logprime = (-lam - 1.0) * log_g - 0.5 * (u * u) + _const_like(u, 0.5 * math.log(2.0 / math.pi))
    return val, logprime


def _extreme_inverse_and_lad(x, tail_param):
    log_inner = torch.log1p(tail_param * x)
    log_g = -log_inner / tail_param
    g = torch.exp(log_g)

    # Optionally do erfcinv in float64 for stability
    use64 = (x.dtype == torch.float32)
    if use64:
        erfcinv_val64 = _stable_erfcinv(g.to(torch.float64), log_g.to(torch.float64))
        erfcinv_val = erfcinv_val64.to(x.dtype)
        lad_summand64 = erfcinv_val64 * erfcinv_val64
        lad_summand = lad_summand64.to(x.dtype)
    else:
        erfcinv_val = _stable_erfcinv(g, log_g)
        lad_summand = erfcinv_val * erfcinv_val

    z = SQRT_2 * erfcinv_val

    lad = (-1 - 1 / tail_param) * log_inner
    lad += lad_summand
    lad += torch.log(x.new_tensor(SQRT_PI / SQRT_2))

    return z, lad


def _ttf_sym_inverse_and_logprime(x, lam):
    """
    TODO: Plot against original implementation for sanity!
    Inverse z = R_lambda^{-1}(x) and log|dz/dx| for symmetric-λ TTF with mu=0, sigma=1.
    Equations:
    inner = 1 + lam * |x|
    g = inner^(-1/lam)
    z = sign(x) * √2 * erfcinv(g)
    log|dz/dx| = (-1 - 1/lam) * log(inner) + erfcinv(g)^2 + log(√π/√2)
    Computes erfcinv via ndtri/series (simple version here).
    """
    s = torch.sign(x)
    u = torch.abs(x)
    inner = 1.0 + lam * u
    log_inner = torch.log1p(lam * u)
    log_g = -log_inner / lam
    g = torch.exp(log_g)

    # Stable erfcinv using ndtri; for very small g, upcast to float64
    def _erfcinv_from_erfc(g):
        # erfcinv(g) = -ndtri(0.5*g) / √2
        return -torch.special.ndtri(0.5 * g) / _const_like(g, math.sqrt(2.0))

    if x.dtype == torch.float32:
        E = _erfcinv_from_erfc(g.to(torch.float64)).to(x.dtype)
    else:
        E = _erfcinv_from_erfc(g)

    z = s * _const_like(x, math.sqrt(2.0)) * E
    logprime = (-1.0 - 1.0 / lam) * torch.log(inner) + (E * E) + _const_like(x, 0.5 * math.log(math.pi / 2.0))
    return z, logprime


def neg_extreme_transform_and_lad(z, tail_param):
    def _small_erfcinv(log_z):
        inner = torch.log(torch.tensor(2 / torch.pi)) - 2 * log_z
        inner -= (torch.log(torch.tensor(2 / torch.pi)) - 2 * log_z).log()
        return inner.pow(0.5) / SQRT_2

    erfc_val = torch.special.erfc(z / SQRT_2)
    g = erfc_val.pow(-tail_param)

    stable_g = g > MIN_ERFC_INV

    erfcinv_val = torch.zeros_like(z)
    erfcinv_val[stable_g] = _erfcinv(g[stable_g])

    log_z = -torch.log(z[~stable_g])
    log_z += -z[~stable_g].square() / 2
    log_z += torch.log(z.new_tensor(SQRT_2 / SQRT_PI))
    log_z *= -tail_param[~stable_g]

    erfcinv_val[~stable_g] = _small_erfcinv(log_z)

    x = -erfcinv_val * 2 / (SQRT_PI * tail_param)

    lad = torch.square(erfcinv_val) - 0.5 * torch.square(z)
    lad += torch.log(z.new_tensor(SQRT_2 / SQRT_PI))
    lad += (-1 - tail_param) * torch.log(erfc_val)

    return x, lad


def _g_transform_and_lad(z, p, kappa, epsilon, delta):
    """Apply alternative gumbel transformation."""
    x = kappa * torch.pow(z.square() + delta, p / 2.0) + epsilon * z
    derivative = epsilon + kappa * p * z * torch.pow(z.square() + delta, p / 2.0 - 1)
    lad = torch.log(torch.abs(derivative))
    return x, lad


def _g_inverse_and_lad(x, p, kappa, epsilon, delta, num_iter: int = 50):
    """Inverse alternative gumbel transform via Newton iterations."""
    z = x / (epsilon + kappa)
    for _ in range(num_iter):
        f = kappa * torch.pow(z.square() + delta, p / 2.0) + epsilon * z - x
        df = epsilon + kappa * p * z * torch.pow(z.square() + delta, p / 2.0 - 1)
        z_next = z - f / df
        if torch.max(torch.abs(z_next - z)) < 1e-8:
            z = z_next
            break
        z = z_next
    lad = -torch.log(torch.abs(epsilon + kappa * p * z * torch.pow(z.square() + delta, p / 2.0 - 1)))
    return z, lad


def _exp_tail_transform_and_lad(x, lam):
    y = (torch.exp(lam * x) - 1.0) / lam
    lad = lam * x
    return y, lad


def _exp_tail_inverse_and_lad(y, lam):
    x = torch.log1p(lam * y) / lam
    lad = -torch.log1p(lam * y)
    return x, lad


def _tail_switch_transform(z, pos_tail, neg_tail, shift, scale):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    heavy_tail = tail_param > 0
    heavy_x, heavy_lad = _extreme_transform_and_lad(
        torch.abs(z[heavy_tail]), tail_param[heavy_tail]
    )
    light_x, light_lad = neg_extreme_transform_and_lad(
        torch.abs(z[~heavy_tail]), tail_param[~heavy_tail]
    )

    lad = torch.zeros_like(z)
    x = torch.zeros_like(z)

    x[heavy_tail] = heavy_x
    x[~heavy_tail] = light_x

    lad[heavy_tail] = heavy_lad
    lad[~heavy_tail] = light_lad

    lad += torch.log(scale)
    return sign * x * scale + shift, lad


def _tail_affine_transform(z, pos_tail, neg_tail, shift, scale):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
    lad = lad + torch.log(scale)
    return sign * x * scale + shift, lad


def _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale):
    # affine
    x = (x - shift) / scale

    # tail transform
    sign = torch.sign(x)
    tail_param = torch.where(x > 0, pos_tail, neg_tail)

    z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))

    lad = lad - torch.log(scale)
    return sign * z, lad


def _tail_forward(z, pos_tail, neg_tail):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
    return sign * x, lad.sum(dim=-1)


def _tail_inverse(x, pos_tail, neg_tail):
    sign = torch.sign(x)
    tail_param = torch.where(x > 0, pos_tail, neg_tail)
    z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
    return sign * z, lad.sum(dim=-1)


def _copula_transform_and_lad(u, tail_param):
    inner = torch.pow(1 - u, -tail_param)
    x = (inner - 1) / tail_param
    lad = (-tail_param - 1) * torch.log(1 - u)
    return x, lad


def _copula_inverse_and_lad(x, tail_param):
    u = 1 - torch.pow(tail_param * x + 1, -1 / tail_param)
    lad = (-1 - 1 / tail_param) * torch.log(tail_param * x + 1)
    return u, lad


def _sinh_asinh_transform_and_lad(z, kurtosis_param):
    x = torch.sinh(torch.arcsinh(z) / kurtosis_param)

    lad = torch.log(torch.cosh(torch.arcsinh(z) / kurtosis_param))
    lad -= torch.log(kurtosis_param)
    lad -= 0.5 * torch.log(torch.square(z) + 1)
    return x, lad


def _sinh_asinh_inverse_and_lad(x, kurtosis_param):
    z = torch.sinh(kurtosis_param * torch.arcsinh(x))

    lad = torch.log(torch.cosh(kurtosis_param * torch.arcsinh(x))) + torch.log(
        kurtosis_param
    )
    lad -= 0.5 * torch.log(torch.square(x) + 1)
    return z, lad


def _asymmetric_scale_transform_and_lad(z, pos_scale, neg_scale):
    sq_plus_1 = (z.square() + 1.0).sqrt()
    a = pos_scale + neg_scale
    b = pos_scale - neg_scale

    pos_x = pos_scale * (sq_plus_1 + z)
    neg_x = neg_scale * (z - sq_plus_1)
    x = 0.5 * (pos_x + neg_x - b)

    lad = torch.log1p((b / a) * (z / sq_plus_1))
    lad -= torch.log(torch.tensor(2.0))
    return x, lad


def _asymmetric_scale_inverse_and_lad(x, pos_scale, neg_scale):
    a = pos_scale + neg_scale
    b = pos_scale - neg_scale
    disc = a**2 - b**2

    z_dash = a * b + 2 * a * x
    term_2 = (a**2 + 4 * b * x + 4 * x**2).sqrt()

    z = (z_dash - torch.sign(b) * term_2) / disc

    lad = torch.log(2 * a - torch.sign(b) * (2 * b + 4 * x) / term_2)
    lad -= torch.log(disc)
    return z, lad


def two_scale_affine_forward(z, shift, scale_neg, scale_pos, bound=torch.tensor(1.0)):
    # build batch x dim x knots arrays
    derivatives = torch.ones([*z.shape, 3])
    derivatives[:, :, 0] = scale_neg
    derivatives[:, :, -1] = scale_pos

    input_knots = torch.zeros([*z.shape, 3])
    input_knots[:, :, 0] = -bound
    input_knots[:, :, -1] = bound

    output_knots = torch.zeros([*z.shape, 3])
    output_knots[:, :, 0] = -bound
    output_knots[:, :, -1] = bound

    neg_region = z < -bound
    pos_region = z > bound
    body = ~torch.logical_or(neg_region, pos_region)
    neg_scale_ix = (neg_region * torch.arange(z.shape[-1]))[neg_region]
    pos_scale_ix = (pos_region * torch.arange(z.shape[-1]))[pos_region]

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    x[neg_region] = (z[neg_region] + bound) * scale_neg[neg_scale_ix] - bound
    x[pos_region] = (z[pos_region] - bound) * scale_pos[pos_scale_ix] + bound
    lad[neg_region] = -torch.log(scale_neg[neg_scale_ix])
    lad[pos_region] = -torch.log(scale_pos[pos_scale_ix])

    body_x, body_lad = forward_rqs(
        z[body], input_knots[body], output_knots[body], derivatives[body]
    )
    x[body] = body_x
    # this has already been inverted, so undo for subsequent inversion
    lad[body] = -body_lad

    x += shift

    return x, lad


def two_scale_affine_inverse(x, shift, scale_neg, scale_pos, bound=torch.tensor(1.0)):
    # build batch x dim x knots arrays
    derivatives = torch.ones([*x.shape, 3])
    derivatives[:, :, 0] = scale_neg
    derivatives[:, :, -1] = scale_pos

    input_knots = torch.zeros([*x.shape, 3])
    input_knots[:, :, 0] = -bound
    input_knots[:, :, -1] = bound

    output_knots = torch.zeros([*x.shape, 3])
    output_knots[:, :, 0] = -bound
    output_knots[:, :, -1] = bound

    # undo shift
    x -= shift

    # regions and place holders
    neg_region = x < -bound
    pos_region = x > bound
    body = ~torch.logical_or(neg_region, pos_region)
    neg_scale_ix = (neg_region * torch.arange(x.shape[-1]))[neg_region]
    pos_scale_ix = (pos_region * torch.arange(x.shape[-1]))[pos_region]

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # scales
    z[neg_region] = (x[neg_region] + bound) / scale_neg[neg_scale_ix] - bound
    z[pos_region] = (x[pos_region] - bound) / scale_pos[pos_scale_ix] + bound
    lad[neg_region] = torch.log(scale_neg[neg_scale_ix])
    lad[pos_region] = torch.log(scale_pos[pos_scale_ix])

    # body
    body_z, body_lad = inverse_rqs(
        x[body], input_knots[body], output_knots[body], derivatives[body]
    )
    z[body] = body_z
    lad[body] = body_lad

    return z, lad


def flip(transform):
    """
    if it is an autoregressive transform change around the element wise transform,
    to preserve the direction of the autoregression. Otherwise, we can flip the full
    transformation.
    """
    if issubclass(type(transform), AutoregressiveTransform):
        _inverse = transform._elementwise_inverse
        transform._elementwise_inverse = transform._elementwise_forward
        transform._elementwise_forward = _inverse
    else:
        _inverse = transform.inverse
        transform.inverse = transform.forward
        transform.forward = _inverse

    return transform



##################################################
# -------- Modified TTF Transformations -------- #
##################################################

# Transforming both sides

def r_lin_both_forward(z, lam_pos, lam_neg, a_pos, a_neg):
    """
    Test and also try with original (sym) TTF implementation
    Piecewise-linear modification with different λ on each side.
    Returns (x, log|dx/dz|).
    """
    # Precompute anchors
    Ra_minus, logRp_minus = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_neg, lam_neg)
    Ra_plus, logRp_plus = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_pos, lam_pos)
    Rp_a_minus = torch.exp(logRp_minus) # R’{λ-}(a-)
    Rp_a_plus = torch.exp(logRp_plus) # R’{λ_+}(a_+)

    # Regions
    left = z <= a_neg
    mid  = (z >= a_neg) & (z <= a_pos)
    right= z >= a_pos

    Rm_z, logRp_z_m = _ttf_sym_value_and_logprime(z, lam_neg)
    Rp_z, logRp_z_p = _ttf_sym_value_and_logprime(z, lam_pos)

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    # Left: (Rλ-(z) - Rλ-(a-)) / Rλ-'(a-) + a-
    x[left] = ((Rm_z[left] - Ra_minus) / Rp_a_minus) + a_neg
    lad[left] = logRp_z_m[left] - logRp_minus  # log(R'(z)/R'(a-))

    # Mid: identity
    x[mid] = z[mid]
    lad[mid] = 0.0

    # Right: (Rλ+(z) - Rλ+(a+)) / Rλ+'(a+) + a+
    x[right] = ((Rp_z[right] - Ra_plus) / Rp_a_plus) + a_pos
    lad[right] = logRp_z_p[right] - logRp_plus

    return x, lad


def r_lin_both_inverse(x, lam_pos, lam_neg, a_pos, a_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_lin_both_forward.
    Returns (z, log|dz/dx|).
    """
    # Precompute anchors
    Ra_minus, logRp_minus = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_neg, lam_neg)
    Ra_plus, logRp_plus = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_pos, lam_pos)
    Rp_a_minus = torch.exp(logRp_minus)
    Rp_a_plus = torch.exp(logRp_plus)

    left = x <= a_neg
    mid  = (x >= a_neg) & (x <= a_pos)
    right= x >= a_pos

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # Left: Rλ-^{-1}( Rλ-(a-) + Rλ-'(a-) * (x - a-) )
    y_left = Ra_minus + Rp_a_minus * (x[left] - a_neg)
    z_left, lad_left = _ttf_sym_inverse_and_logprime(y_left, lam_neg)
    z[left] = z_left
    lad[left] = torch.log(Rp_a_minus) + lad_left  # log(R'(a-)) + log|(R^{-1})'|

    # Mid: identity
    z[mid] = x[mid]
    lad[mid] = 0.0

    # Right: Rλ+^{-1}( Rλ+(a+) + Rλ+'(a+) * (x - a+) )
    y_right = Ra_plus + Rp_a_plus * (x[right] - a_pos)
    z_right, lad_right = _ttf_sym_inverse_and_logprime(y_right, lam_pos)
    z[right] = z_right
    lad[right] = torch.log(Rp_a_plus) + lad_right

    return z, lad


def r_erfi_both_forward(z, lam_pos, lam_neg, a_pos, a_neg, r_pos, t_pos, r_neg, t_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation and original erfi implementation
    erfi-smoothing modification on both sides with continuity and C1 at 0 using beta = r_-/r_+.
    Returns (x, log|dx/dz|).
    Inputs r±>0, t±>0, a_-<0<a_+.
    """
    beta_mpm = r_neg / r_pos

    # Constants c_- and c_+
    sqrt_pi_over_2 = _const_like(z, math.sqrt(math.pi) / 2.0)
    c_minus = sqrt_pi_over_2 * (r_neg / torch.sqrt(t_neg)) * _erfi_complex(torch.sqrt(t_neg) * a_neg)
    c_plus  = beta_mpm * sqrt_pi_over_2 * (r_pos / torch.sqrt(t_pos)) * _erfi_complex(torch.sqrt(t_pos) * a_pos)

    left_tail = z <= a_neg
    left_mid  = (z >= a_neg) & (z <= 0)
    right_mid = (z >= 0) & (z <= a_pos)
    right_tail= z >= a_pos

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    # Left tail: Rλ-(z) - Rλ-(a-) + c_-
    Rm_z, logRp_z_m = _ttf_sym_value_and_logprime(z, lam_neg)
    Ra_minus, _ = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_neg, lam_neg)
    x[left_tail] = (Rm_z[left_tail] - Ra_minus) + c_minus
    lad[left_tail] = logRp_z_m[left_tail]

    # Left mid: (√π/2) r_- / √t_- erfi(√t_- z)
    x[left_mid] = sqrt_pi_over_2 * (r_neg / torch.sqrt(t_neg)) * _erfi_complex(torch.sqrt(t_neg) * z[left_mid])
    lad[left_mid] = torch.log(r_neg) + (t_neg * z[left_mid] * z[left_mid])

    # Right mid: beta * (√π/2) r_+ / √t_+ erfi(√t_+ z)
    x[right_mid] = beta_mpm * sqrt_pi_over_2 * (r_pos / torch.sqrt(t_pos)) * _erfi_complex(torch.sqrt(t_pos) * z[right_mid])
    lad[right_mid] = torch.log(beta_mpm) + torch.log(r_pos) + (t_pos * z[right_mid] * z[right_mid])

    # Right tail: beta * (Rλ+(z) - Rλ+(a+)) + c_+
    Rp_z, logRp_z_p = _ttf_sym_value_and_logprime(z, lam_pos)
    Ra_plus, _ = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_pos, lam_pos)
    x[right_tail] = beta_mpm * (Rp_z[right_tail] - Ra_plus) + c_plus
    lad[right_tail] = torch.log(beta_mpm) + logRp_z_p[right_tail]

    return x, lad


def r_erfi_both_inverse(x, lam_pos, lam_neg, a_pos, a_neg, r_pos, t_pos, r_neg, t_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation and original erfi implementation
    Inverse of r_erfi_both_forward.
    Returns (z, log|dz/dx|).
    """
    beta_mpm = r_neg / r_pos
    beta_pm = 1.0 / beta_mpm
    sqrt_pi_over_2 = _const_like(x, math.sqrt(math.pi) / 2.0)

    c_minus = sqrt_pi_over_2 * (r_neg / torch.sqrt(t_neg)) * _erfi_complex(torch.sqrt(t_neg) * a_neg)
    c_plus  = beta_mpm * sqrt_pi_over_2 * (r_pos / torch.sqrt(t_pos)) * _erfi_complex(torch.sqrt(t_pos) * a_pos)

    left_tail = x <= c_minus
    left_mid  = (x >= c_minus) & (x <= 0)
    right_mid = (x >= 0) & (x <= c_plus)
    right_tail= x >= c_plus

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # Left tail: Rλ-^{-1}( x - c_- + Rλ-(a-) )
    Ra_minus, _ = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_neg, lam_neg)
    y_left = x[left_tail] - c_minus + Ra_minus
    z_left, lad_left = _ttf_sym_inverse_and_logprime(y_left, lam_neg)
    z[left_tail] = z_left
    lad[left_tail] = lad_left

    # Left mid: (1/√t_-) erfi^{-1}( (2/√π) √t_-/r_- * x )
    arg_left = (2.0 / math.sqrt(math.pi)) * (torch.sqrt(t_neg) / r_neg) * x[left_mid]
    u_left = _erfi_inv(arg_left)
    z[left_mid] = u_left / torch.sqrt(t_neg)
    # dz/dx = exp(-u^2) / r_-
    lad[left_mid] = -torch.log(r_neg) - (u_left * u_left)

    # Right mid: (1/√t_+) erfi^{-1}( (2/√π) √t_+/r_+ * beta_pm * x )
    arg_right = (2.0 / math.sqrt(math.pi)) * (torch.sqrt(t_pos) / r_pos) * (beta_pm * x[right_mid])
    u_right = _erfi_inv(arg_right)
    z[right_mid] = u_right / torch.sqrt(t_pos)
    # dz/dx = beta_pm * exp(-u^2) / r_+
    lad[right_mid] = torch.log(beta_pm) - torch.log(r_pos) - (u_right * u_right)

    # Right tail: Rλ+^{-1}( beta_pm * (x - c_+) + Rλ+(a+) )
    Ra_plus, _ = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_pos, lam_pos)
    y_right = beta_pm * (x[right_tail] - c_plus) + Ra_plus
    z_right, lad_right = _ttf_sym_inverse_and_logprime(y_right, lam_pos)
    z[right_tail] = z_right
    lad[right_tail] = torch.log(beta_pm) + lad_right

    return z, lad


def r_qua_both_forward(z, lam_pos, lam_neg, a_pos, a_neg, c0_pos, c2_pos, c0_neg, c2_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Quadratic-slope (cubic primitive) smoothing on both sides with C1 at 0 using beta = c0_-/c0_+.
    Returns (x, log|dx/dz|).
    """
    beta_mpm = c0_neg / c0_pos
    c_minus = (1.0 / 3.0) * c2_neg * (a_neg ** 3) + c0_neg * a_neg
    c_plus = beta_mpm * ((1.0 / 3.0) * c2_pos * (a_pos ** 3) + c0_pos * a_pos)

    left_tail = z <= a_neg
    left_mid  = (z >= a_neg) & (z <= 0)
    right_mid = (z >= 0) & (z <= a_pos)
    right_tail= z >= a_pos

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    # Left tail
    Rm_z, logRp_z_m = _ttf_sym_value_and_logprime(z, lam_neg)
    Ra_minus, _ = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_neg, lam_neg)
    x[left_tail] = (Rm_z[left_tail] - Ra_minus) + c_minus
    lad[left_tail] = logRp_z_m[left_tail]

    # Left mid: (1/3) c2^- z^3 + c0^- z
    x[left_mid] = (1.0 / 3.0) * c2_neg * (z[left_mid] ** 3) + c0_neg * z[left_mid]
    lad[left_mid] = torch.log(c2_neg * (z[left_mid] ** 2) + c0_neg)

    # Right mid: beta * ((1/3) c2^+ z^3 + c0^+ z)
    x[right_mid] = beta_mpm * ((1.0 / 3.0) * c2_pos * (z[right_mid] ** 3) + c0_pos * z[right_mid])
    lad[right_mid] = torch.log(beta_mpm) + torch.log(c2_pos * (z[right_mid] ** 2) + c0_pos)

    # Right tail
    Rp_z, logRp_z_p = _ttf_sym_value_and_logprime(z, lam_pos)
    Ra_plus, _ = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_pos, lam_pos)
    x[right_tail] = beta_mpm * (Rp_z[right_tail] - Ra_plus) + c_plus
    lad[right_tail] = torch.log(beta_mpm) + logRp_z_p[right_tail]

    return x, lad


def r_qua_both_inverse(x, lam_pos, lam_neg, a_pos, a_neg, c0_pos, c2_pos, c0_neg, c2_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_qua_both_forward.
    Returns (z, log|dz/dx|).
    """
    beta_mpm = c0_neg / c0_pos
    beta_pm = 1.0 / beta_mpm
    c_minus = (1.0 / 3.0) * c2_neg * (a_neg ** 3) + c0_neg * a_neg
    c_plus = beta_mpm * ((1.0 / 3.0) * c2_pos * (a_pos ** 3) + c0_pos * a_pos)

    left_tail = x <= c_minus
    left_mid  = (x >= c_minus) & (x <= 0)
    right_mid = (x >= 0) & (x <= c_plus)
    right_tail= x >= c_plus

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # Left tail: Rλ-^{-1}(x - c_- + Rλ-(a-))
    Ra_minus, _ = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_neg, lam_neg)
    y_left = x[left_tail] - c_minus + Ra_minus
    z_left, lad_left = _ttf_sym_inverse_and_logprime(y_left, lam_neg)
    z[left_tail] = z_left
    lad[left_tail] = lad_left

    # Left mid: Cardano with p = 3 c0^- / c2^-, q = -3 x / c2^-
    pL = (3.0 * c0_neg) / c2_neg
    qL = -3.0 * x[left_mid] / c2_neg
    deltaL = (qL / 2.0) ** 2 + (pL / 3.0) ** 3
    zL = _cbrt(-qL / 2.0 + torch.sqrt(deltaL)) + _cbrt(-qL / 2.0 - torch.sqrt(deltaL))
    z[left_mid] = zL
    lad[left_mid] = -torch.log(c2_neg * (zL ** 2) + c0_neg)

    # Right mid: Cardano with p = 3 c0^+ / c2^+, q = -3 beta_pm x / c2^+
    pR = (3.0 * c0_pos) / c2_pos
    qR = -3.0 * (beta_pm * x[right_mid]) / c2_pos
    deltaR = (qR / 2.0) ** 2 + (pR / 3.0) ** 3
    zR = _cbrt(-qR / 2.0 + torch.sqrt(deltaR)) + _cbrt(-qR / 2.0 - torch.sqrt(deltaR))
    z[right_mid] = zR
    lad[right_mid] = torch.log(beta_pm) - torch.log(c2_pos * (zR ** 2) + c0_pos)

    # Right tail: Rλ+^{-1}( beta_pm * (x - c_+) + Rλ+(a+) )
    Ra_plus, _ = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_pos, lam_pos)
    y_right = beta_pm * (x[right_tail] - c_plus) + Ra_plus
    z_right, lad_right = _ttf_sym_inverse_and_logprime(y_right, lam_pos)
    z[right_tail] = z_right
    lad[right_tail] = torch.log(beta_pm) + lad_right

    return z, lad


# Transforming only one side

def r_right_forward(z, lam_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Transform only the right tail with basic TTF; left side is linear with slope sqrt(2/pi).
    Returns (x, log|dx/dz|).
    """
    slope = _const_like(z, math.sqrt(2.0 / math.pi))
    left = z <= 0
    right = z >= 0

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    x[left] = slope * z[left]
    lad[left] = torch.log(slope)

    Rp_z, logRp_z_p = _ttf_sym_value_and_logprime(z, lam_pos)
    x[right] = Rp_z[right]
    lad[right] = logRp_z_p[right]

    return x, lad


def r_left_forward(z, lam_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Transform only the left tail with basic TTF; right side is linear with slope sqrt(2/pi).
    Returns (x, log|dx/dz|).
    """
    val_at_minus_z, lad_at_minus_z = r_right_forward(-z, lam_neg)
    return - val_at_minus_z, lad_at_minus_z


def r_right_inverse(x, lam_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_right_forward.
    Returns (z, log|dz/dx|).
    """
    slope_inv = _const_like(x, math.sqrt(math.pi / 2.0))
    left = x <= 0
    right = x >= 0

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    z[left] = slope_inv * x[left]
    lad[left] = torch.log(slope_inv)

    z_right, lad_right = _ttf_sym_inverse_and_logprime(x[right], lam_pos)
    z[right] = z_right
    lad[right] = lad_right

    return z, lad


def r_left_inverse(x, lam_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_left_forward.
    Returns (z, log|dz/dx|).
    """
    val_at_minus_x, lad_at_minus_x = r_right_inverse(-x, lam_neg)
    return - val_at_minus_x, lad_at_minus_x


def r_lin_right_forward(z, lam_pos, a_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Piecewise-linear modification transforming only the right tail.
    Returns (x, log|dx/dz|).
    """
    Rp_a, logRp_a = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_pos, lam_pos)
    Rp_a_val = torch.exp(logRp_a)

    left = z <= a_pos
    right= z >= a_pos

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    x[left] = z[left]
    lad[left] = 0.0

    Rp_z, logRp_z = _ttf_sym_value_and_logprime(z, lam_pos)
    x[right] = ((Rp_z[right] - Rp_a) / Rp_a_val) + a_pos
    lad[right] = logRp_z[right] - logRp_a

    return x, lad


def r_lin_left_forward(z, lam_neg, a_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Piecewise-linear modification transforming only the left tail.
    Returns (x, log|dx/dz|).
    """
    assert a_neg <= 0.0, "a_neg must be negative!"
    val_at_minus_z, lad_at_minus_z = r_lin_right_forward(-z, lam_neg, -a_neg)
    return - val_at_minus_z, lad_at_minus_z


def r_lin_right_inverse(x, lam_pos, a_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_lin_right_forward.
    Returns (z, log|dz/dx|).
    """
    Rp_a, logRp_a = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_pos, lam_pos)
    Rp_a_val = torch.exp(logRp_a)

    left = x <= a_pos
    right= x >= a_pos

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    z[left] = x[left]
    lad[left] = 0.0

    y = Rp_a + Rp_a_val * (x[right] - a_pos)
    z_right, lad_right = _ttf_sym_inverse_and_logprime(y, lam_pos)
    z[right] = z_right
    lad[right] = torch.log(Rp_a_val) + lad_right

    return z, lad


def r_lin_left_inverse(x, lam_neg, a_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_lin_left_forward.
    Returns (z, log|dz/dx|).
    """
    assert a_neg <= 0.0, "a_neg must be negative!"
    val_at_minus_x, lad_at_minus_x = r_lin_right_inverse(-x, lam_neg, -a_neg)
    return - val_at_minus_x, lad_at_minus_x



def r_erfi_right_forward(z, lam_pos, a_pos, r_pos, t_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation and original erfi implementation
    erfi-smoothing transforming only the right tail; left side is linear with slope r_+ for being C1 at 0.
    Returns (x, log|dx/dz|).
    """
    sqrt_pi_over_2 = _const_like(z, math.sqrt(math.pi) / 2.0)
    c_plus = sqrt_pi_over_2 * (r_pos / torch.sqrt(t_pos)) * _erfi_complex(torch.sqrt(t_pos) * a_pos)

    left = z <= 0
    mid  = (z >= 0) & (z <= a_pos)
    right= z >= a_pos

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    # Left: r_+ z
    x[left] = r_pos * z[left]
    lad[left] = torch.log(r_pos)

    # Mid: (√π/2) r_+ / √t_+ erfi(√t_+ z)
    x[mid] = sqrt_pi_over_2 * (r_pos / torch.sqrt(t_pos)) * _erfi_complex(torch.sqrt(t_pos) * z[mid])
    lad[mid] = torch.log(r_pos) + (t_pos * z[mid] * z[mid])

    # Right: Rλ+(z) - Rλ+(a+) + c_+
    Rp_z, logRp_z = _ttf_sym_value_and_logprime(z, lam_pos)
    Rp_a, _ = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_pos, lam_pos)
    x[right] = (Rp_z[right] - Rp_a) + c_plus
    lad[right] = logRp_z[right]

    return x, lad


def r_erfi_left_forward(z, lam_neg, a_neg, r_neg, t_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation and original erfi implementation
    erfi-smoothing transforming only the left tail; righ side is linear with slope r_- for being C1 at 0.
    Returns (x, log|dx/dz|).
    """
    assert a_neg <= 0.0, "a_neg must be negative!"
    assert r_neg > 0 and t_neg > 0, "r_neg and t_neg must be positive!"
    val_at_minus_z, lad_at_minus_z = r_erfi_right_forward(-z, lam_neg, -a_neg, r_neg, t_neg)
    return - val_at_minus_z, lad_at_minus_z


def r_erfi_right_inverse(x, lam_pos, a_pos, r_pos, t_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation and original erfi implementation
    Inverse of r_erfi_right_forward.
    Returns (z, log|dz/dx|).
    """
    sqrt_pi_over_2 = _const_like(x, math.sqrt(math.pi) / 2.0)
    c_plus = sqrt_pi_over_2 * (r_pos / torch.sqrt(t_pos)) * _erfi_complex(torch.sqrt(t_pos) * a_pos)

    left = x <= 0
    mid  = (x >= 0) & (x <= c_plus)
    right= x >= c_plus

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # Left: x / r_+
    z[left] = x[left] / r_pos
    lad[left] = -torch.log(r_pos)

    # Mid: (1/√t_+) erfi^{-1}( (2/√π) √t_+/r_+ x )
    arg = (2.0 / math.sqrt(math.pi)) * (torch.sqrt(t_pos) / r_pos) * x[mid]
    u = _erfi_inv(arg)
    z[mid] = u / torch.sqrt(t_pos)
    lad[mid] = -torch.log(r_pos) - (u * u)  # dz/dx = exp(-u^2)/r_+

    # Right: Rλ+^{-1}( x - c_+ + Rλ+(a+) )
    Rp_a, _ = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_pos, lam_pos)
    y = x[right] - c_plus + Rp_a
    z_right, lad_right = _ttf_sym_inverse_and_logprime(y, lam_pos)
    z[right] = z_right
    lad[right] = lad_right

    return z, lad


def r_erfi_left_inverse(x, lam_neg, a_neg, r_neg, t_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation and original erfi implementation
    Inverse of r_erfi_left_forward.
    Returns (z, log|dz/dx|).
    """
    assert a_neg <= 0.0, "a_neg must be negative!"
    assert r_neg > 0 and t_neg > 0, "r_neg and t_neg must be positive!"
    val_at_minus_x, lad_at_minus_x = r_erfi_right_inverse(-x, lam_neg, -a_neg, r_neg, t_neg)
    return - val_at_minus_x, lad_at_minus_x


def r_qua_right_forward(z, lam_pos, a_pos, c0_pos, c2_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Quadratic-slope smoothing transforming only the right tail; left side has slope c0^+ for being C1 at 0.
    Returns (x, log|dx/dz|).
    """
    c_plus = (1.0 / 3.0) * c2_pos * (a_pos ** 3) + c0_pos * a_pos

    left = z <= 0
    mid  = (z >= 0) & (z <= a_pos)
    right= z >= a_pos

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    # Left: c0^+ z
    x[left] = c0_pos * z[left]
    lad[left] = torch.log(c0_pos)

    # Mid: (1/3) c2^+ z^3 + c0^+ z
    x[mid] = (1.0 / 3.0) * c2_pos * (z[mid] ** 3) + c0_pos * z[mid]
    lad[mid] = torch.log(c2_pos * (z[mid] ** 2) + c0_pos)

    # Right: Rλ+(z) - Rλ+(a+) + c_+
    Rp_z, logRp_z = _ttf_sym_value_and_logprime(z, lam_pos)
    Rp_a, _ = _ttf_sym_value_and_logprime(_const_like(z, 0.0) + a_pos, lam_pos)
    x[right] = (Rp_z[right] - Rp_a) + c_plus
    lad[right] = logRp_z[right]

    return x, lad


def r_qua_left_forward(z, lam_neg, a_neg, c0_neg, c2_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Quadratic-slope smoothing transforming only the left tail; right side has slope c0^- for being C1 at 0.
    Returns (x, log|dx/dz|).
    """
    assert a_neg <= 0.0, "a_neg must be negative!"
    assert c0_neg > 0 and c2_neg > 0, "c0_neg and c2_neg must be positive!"
    val_at_minus_z, lad_at_minus_z = r_qua_right_forward(-z, lam_neg, -a_neg, c0_neg, c2_neg)
    return -val_at_minus_z, lad_at_minus_z


def r_qua_right_inverse(x, lam_pos, a_pos, c0_pos, c2_pos):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_qua_right_forward.
    Returns (z, log|dz/dx|).
    """
    c_plus = (1.0 / 3.0) * c2_pos * (a_pos ** 3) + c0_pos * a_pos

    left = x <= 0
    mid  = (x >= 0) & (x <= c_plus)
    right= x >= c_plus

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # Left: x / c0^+
    z[left] = x[left] / c0_pos
    lad[left] = -torch.log(c0_pos)

    # Mid: Cardano with p = 3 c0^+ / c2^+, q = -3 x / c2^+
    p = (3.0 * c0_pos) / c2_pos
    q = -3.0 * x[mid] / c2_pos
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3
    zM = _cbrt(-q / 2.0 + torch.sqrt(delta)) + _cbrt(-q / 2.0 - torch.sqrt(delta))
    z[mid] = zM
    lad[mid] = -torch.log(c2_pos * (zM ** 2) + c0_pos)

    # Right: Rλ+^{-1}( x - c_+ + Rλ+(a+) )
    Rp_a, _ = _ttf_sym_value_and_logprime(_const_like(x, 0.0) + a_pos, lam_pos)
    y = x[right] - c_plus + Rp_a
    z_right, lad_right = _ttf_sym_inverse_and_logprime(y, lam_pos)
    z[right] = z_right
    lad[right] = lad_right

    return z, lad


def r_qua_left_inverse(x, lam_neg, a_neg, c0_neg, c2_neg):
    """
    TODO: Test and also try with original (sym) TTF implementation
    Inverse of r_qua_right_forward.
    Returns (z, log|dz/dx|).
    """
    assert a_neg <= 0.0, "a_neg must be negative!"
    assert c0_neg > 0 and c2_neg > 0, "c0_neg and c2_neg must be positive!"
    val_at_minus_x, lad_at_minus_x = r_qua_right_inverse(-x, lam_neg, -a_neg, c0_neg, c2_neg)
    return -val_at_minus_x, lad_at_minus_x



####################################################################
# ----- Transformation classes implementing the above trafos ----- #
####################################################################



class TailMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
    ):
        self.features = features

        super(TailMarginalTransform, self).__init__()

        # init with heavy tail, otherwise heavy targets may fail to fit
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape

        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))

    def forward(self, z, context=None):
        pos_tail_param = softplus(self._unc_pos_tail)
        neg_tail_param = softplus(self._unc_neg_tail)
        x, lad = _tail_forward(z, pos_tail_param, neg_tail_param)
        return x, lad

    def inverse(self, x, context=None):
        pos_tail_param = softplus(self._unc_pos_tail)
        neg_tail_param = softplus(self._unc_neg_tail)
        z, lad = _tail_inverse(x, pos_tail_param, neg_tail_param)
        return z, lad

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False


class AffineMarginalTransform(Transform):
    def __init__(
        self,
        features,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(AffineMarginalTransform, self).__init__()

        # random inits if needed
        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    def forward(self, z, context=None):
        shift = self._unc_shift
        # scale = softplus(self._unc_scale)
        scale = 1e-3 + softplus(self._unc_scale)

        x = z * scale + shift
        lad = torch.log(scale).sum()
        return x, lad

    def inverse(self, x, context=None):
        """heavy -> light"""
        shift = self._unc_shift
        scale = 1e-3 + softplus(self._unc_scale)

        z = (x - shift) / scale
        lad = -torch.log(scale).sum()
        return z, lad


class RQSMarginalTransform(Transform):
    def __init__(
        self,
        features,
        num_bins=10,
        tail_bound=1.0,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        self.features = features

        super(RQSMarginalTransform, self).__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = "linear"
        self.tail_bound = tail_bound
        self.spline_fn = (
            unconstrained_rational_quadratic_spline_forward,
            unconstrained_rational_quadratic_spline_inverse,
        )
        self.spline_kwargs = {
            "tails": self.tails,
            "left_tail_bound": -self.tail_bound,
            "right_tail_bound": self.tail_bound,
        }
        self.param_dim = self.num_bins * 3 - 1  # per dim
        self._unc_widths = torch.zeros([1, self.features, self.num_bins])
        self._unc_heights = torch.zeros([1, self.features, self.num_bins])
        self._unc_derivatives = torch.zeros([1, self.features, self.num_bins - 1])

    def forward(self, inputs, context=None):
        batch_dim = inputs.shape[0]
        outputs, logabsdet = self.spline_fn[0](
            inputs=inputs,
            unnormalized_widths=self._unc_widths.repeat(batch_dim, 1, 1),
            unnormalized_heights=self._unc_heights.repeat(batch_dim, 1, 1),
            unnormalized_derivatives=self._unc_derivatives.repeat(batch_dim, 1, 1),
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **self.spline_kwargs,
        )

        return outputs, logabsdet.sum(dim=-1)

    def inverse(self, inputs, context=None):
        batch_dim = inputs.shape[0]
        outputs, logabsdet = self.spline_fn[1](
            inputs=inputs,
            unnormalized_widths=self._unc_widths.repeat(batch_dim, 1, 1),
            unnormalized_heights=self._unc_heights.repeat(batch_dim, 1, 1),
            unnormalized_derivatives=self._unc_derivatives.repeat(batch_dim, 1, 1),
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **self.spline_kwargs,
        )

        return outputs, logabsdet.sum(dim=-1)


class TailAffineMarginalTransform(Transform):
    # TODO: Wrap modified TTF transformations here and test!
    # NOTE: Remember that the modified TTF versions were implemented without shift and scale, so these parameters must be implemented here!
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(TailAffineMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    @property
    def pos_tail(self):
        return softplus(self._unc_pos_tail)

    @property
    def neg_tail(self):
        return softplus(self._unc_neg_tail)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def forward(self, z, context=None):
        """light -> heavy"""
        x, lad = _tail_affine_transform(
            z, self.pos_tail, self.neg_tail, self.shift, self.scale
        )
        return x, lad.sum(dim=-1)

    def inverse(self, x, context=None):
        """heavy -> light"""
        z, lad = _tail_affine_inverse(
            x, self.pos_tail, self.neg_tail, self.shift, self.scale
        )
        return z, lad.sum(dim=-1)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False


class AsymmetricTailAffineMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(AsymmetricTailAffineMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    @property
    def pos_tail(self):
        return softplus(self._unc_pos_tail)

    @property
    def neg_tail(self):
        return softplus(self._unc_neg_tail)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def inverse(self, z, context=None):
        """light -> heavy"""
        
        # tail transform
        rescale = z.new_tensor(SQRT_PI / SQRT_2) # ensures agreement on lad at the origin 
        pos_x, pos_lad = _extreme_inverse_and_lad(torch.abs(z) / rescale, self.pos_tail)
        neg_x, neg_lad = _extreme_transform_and_lad(torch.abs(z) * rescale, self.neg_tail)
        neg_x = -neg_x

        x = torch.where(z > 0, pos_x, neg_x)
        lad = torch.where(
            z > 0, 
            pos_lad - torch.log(rescale), 
            neg_lad + torch.log(rescale)
        )

        # affine
        x = x * self.scale + self.shift

        lad += torch.log(self.scale)
        return z, lad.sum(dim=-1)

    def forward(self, x, context=None):
        """
        Data -> Noise
        +tail heavy -> light
        -tail light -> heavy
        """
        # affine
        x = (x - self.shift) / self.scale

        # tail transform
        rescale = x.new_tensor(SQRT_PI / SQRT_2) # ensures agreement on lad at the origin 
        pos_z, pos_lad = _extreme_inverse_and_lad(torch.abs(x) / rescale, self.pos_tail)
        neg_z, neg_lad = _extreme_transform_and_lad(torch.abs(x) * rescale, self.neg_tail)
        neg_z = -neg_z

        z = torch.where(x > 0, pos_z, neg_z)
        lad = torch.where(
            z > 0, 
            pos_lad - torch.log(rescale), 
            neg_lad + torch.log(rescale)
        )

        lad -= torch.log(self.scale)

        return z, lad.sum(dim=-1)


class TailSwitchMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(TailSwitchMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._pos_tail = torch.nn.parameter.Parameter(pos_tail_init)
        self._neg_tail = torch.nn.parameter.Parameter(neg_tail_init)
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init - 1e-3))

    @staticmethod
    def _tail_parameterisation(z):

        out, _ = torch.vmap(
            univariate_forward_rqs, in_dims=(0, None, None, None), out_dims=(0, 0)
        )(
            z.reshape(-1, 1).clamp(min=-0.5 + 1e-6, max=0.5 - 1e-6),
            torch.tensor([[-0.5, 0.0, 0.5]]),
            torch.tensor([[-0.5, 0.0, 0.5]]),
            torch.tensor([[1.0, 0.2, 1.0]]),
        )

        out = torch.where(
            z.abs() < 0.5,
            out.reshape(-1),
            z,
        )
        return out

    def forward(self, x, context=None):
        # affine
        x = (x - self.shift) / self.scale

        sign = torch.sign(x)
        tail_param = torch.where(x > 0, self.pos_tail, self.neg_tail)

        # negative tail param as this is being applied in data -> noise
        # so data has tail_param tail
        z, lad = TailSwitchMarginalTransform._transformation(torch.abs(x), -tail_param)
        lad -= torch.log(self.scale)
        return sign * z, lad.sum(dim=-1)

    def inverse(self, z, context=None):
        sign = torch.sign(z)
        tail_param = torch.where(
            z > 0,
            self.pos_tail,
            self.neg_tail,
        )
        x, lad = TailSwitchMarginalTransform._transformation(torch.abs(z), tail_param)
        lad += torch.log(self.scale)
        x = sign * x * self.scale + self.shift
        return x, lad.sum(dim=-1)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    @property
    def pos_tail(self):
        return self._tail_parameterisation(self._pos_tail)

    @property
    def neg_tail(self):
        return self._tail_parameterisation(self._neg_tail)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False

    @staticmethod
    def extreme_transform(z, tau):
        # tau > 1
        tail_param = (tau - 1).abs()
        g = torch.special.erfc(z / SQRT_2)
        x = (torch.pow(g, -tail_param) - 1) / tail_param
        x *= SQRT_PI / SQRT_2

        lad = torch.log(g) * (-tail_param - 1)
        lad -= 0.5 * torch.square(z)

        return x, lad

    @staticmethod
    def asymp_h_transform(z):
        g = torch.special.erfc(z / SQRT_2)
        return -torch.log(g) * SQRT_PI / SQRT_2

    @staticmethod
    def inter_h_transform(z, tau):
        # tau in [0, 1]
        tail_param = tau - 1
        g = torch.special.erfc(z / SQRT_2)
        x = -_erfcinv(g.pow(-tail_param))
        x *= SQRT_2 / tail_param

        lad = torch.log(g) * (-tail_param - 1)
        lad -= 0.5 * torch.square(z)
        lad += 0.5 * tail_param.square() * x.square()

        return x, lad

    @staticmethod
    def inter_l_transform(z, tau):
        # tau in [-1, 0]
        tail_param = -tau - 1  # tail_param = 1 - (tau - 1)
        inner = torch.special.erfc(-tail_param * z / SQRT_2)

        g = torch.pow(inner, -1 / tail_param)
        log_g = -torch.log(inner) / tail_param

        erfcinv_val = _stable_erfcinv(g, log_g)

        x = SQRT_2 * erfcinv_val

        lad = -0.5 * tail_param.square() * z.square()
        lad += (-1 - 1 / tail_param) * torch.log(inner)
        lad += torch.square(erfcinv_val)

        return x, lad

    @staticmethod
    def asymp_l_transform(z):
        g = torch.exp(-(SQRT_2 / SQRT_PI) * z)
        return SQRT_2 * _erfcinv(g)

    @staticmethod
    def extreme_inverse(z, tau):
        # tau < -1
        tail_param = -tau - 1

        inner = 1 + tail_param * (SQRT_2 / SQRT_PI) * z

        g = torch.pow(inner, -1 / tail_param)
        log_g = -torch.log(inner) / tail_param

        erfcinv_val = _stable_erfcinv(g, log_g)
        x = SQRT_2 * erfcinv_val

        lad = (-1 - 1 / tail_param) * torch.log(inner)
        lad += torch.square(erfcinv_val)

        return x, lad

    @staticmethod
    def _transformation(z, tau):
        if tau.shape[0] == 1:  # fixed parameter for each observation
            tau = tau.repeat((z.shape[0], 1))

        assert (
            z.shape[1] == tau.shape[1]
        ), f"Tail parameter must be 2D [1, {z.shape[1]}] or {z.shape[1]}"

        heavy_tail = tau > 1
        light_tail = tau < -1
        iheavy = torch.logical_and(~heavy_tail, ~light_tail)
        iheavy = torch.logical_and(iheavy, tau >= 0)
        ilight = torch.logical_and(~heavy_tail, ~light_tail)
        ilight = torch.logical_and(ilight, tau < 0)

        heavy_x, heavy_lad = TailSwitchMarginalTransform.extreme_transform(
            z[heavy_tail], tau[heavy_tail]
        )
        iheavy_x, iheavy_lad = TailSwitchMarginalTransform.inter_h_transform(
            z[iheavy], tau[iheavy]
        )
        ilight_x, ilight_lad = TailSwitchMarginalTransform.inter_l_transform(
            z[ilight], tau[ilight]
        )
        light_x, light_lad = TailSwitchMarginalTransform.extreme_inverse(
            z[light_tail], tau[light_tail]
        )

        x = torch.ones_like(z)
        lad = torch.ones_like(z)

        for index, x_val, lad_val in (
            (heavy_tail, heavy_x, heavy_lad),
            (iheavy, iheavy_x, iheavy_lad),
            (ilight, ilight_x, ilight_lad),
            (light_tail, light_x, light_lad),
        ):
            x[index] = x_val
            lad[index] = lad_val
        return x, lad


class SmoothTailSwitchMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(SmoothTailSwitchMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self.pos_tail = torch.nn.parameter.Parameter(pos_tail_init)
        self.neg_tail = torch.nn.parameter.Parameter(neg_tail_init)
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init - 1e-3))

        self._tail_forward = torch.vmap(
            SmoothTailSwitchMarginalTransform.univariate_smooth_ex_transform,
            in_dims=(1, 0, 0),
            out_dims=(1, 1),
        )

        self._tail_inverse = torch.vmap(
            SmoothTailSwitchMarginalTransform.univariate_smooth_ex_inverse,
            in_dims=(1, 0, 0),
            out_dims=(1, 1),
        )

    def forward(self, x, context=None):
        # affine
        x = (x - self.shift) / self.scale
        z, lad = self._tail_inverse(x, self.pos_tail, self.neg_tail)
        lad -= torch.log(self.scale)
        return z, lad.sum(dim=-1)

    def inverse(self, z, context=None):
        x, lad = self._tail_forward(z, self.pos_tail, self.neg_tail)
        lad += torch.log(self.scale)
        x = x * self.scale + self.shift
        return x, lad.sum(dim=-1)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False

    @staticmethod
    def _real_transformation(z, pos_tau, neg_tau):
        tau = torch.where(z > 0, pos_tau, neg_tau)
        x, lad = SmoothTailSwitchMarginalTransform._tail_transformation(z.abs(), tau)
        return x * z.sign(), lad

    @staticmethod
    def _tail_transformation(z, tau):
        heavy_tail = tau > 1
        light_tail = tau < -1
        iheavy = torch.logical_and(~heavy_tail, ~light_tail)
        iheavy = torch.logical_and(iheavy, tau >= 0)
        ilight = torch.logical_and(~heavy_tail, ~light_tail)
        ilight = torch.logical_and(ilight, tau < 0)

        heavy_x, heavy_lad = TailSwitchMarginalTransform.extreme_transform(
            z, torch.clamp(tau, min=1.0 + 1e-6, max=None)
        )
        iheavy_x, iheavy_lad = TailSwitchMarginalTransform.inter_h_transform(
            z, torch.clamp(tau, min=0.0 - 1e-6, max=1.0 + 1e-6)
        )
        ilight_x, ilight_lad = TailSwitchMarginalTransform.inter_l_transform(
            z, torch.clamp(tau, min=-1.0 - 1e-6, max=0.0)
        )
        light_x, light_lad = TailSwitchMarginalTransform.extreme_inverse(
            z, torch.clamp(tau, min=None, max=-1.0 - 1e-6)
        )

        x = torch.ones_like(z)
        lad = torch.ones_like(z)

        for index, x_val, lad_val in (
            (heavy_tail, heavy_x, heavy_lad),
            (iheavy, iheavy_x, iheavy_lad),
            (ilight, ilight_x, ilight_lad),
            (light_tail, light_x, light_lad),
        ):
            x = torch.where(index, x_val, x)
            lad = torch.where(index, lad_val, lad)

        return x, lad

    @staticmethod
    def univariate_smooth_ex_transform(z, pos_tau, neg_tau):
        knot_z = torch.tensor([-1.6, 1.6]).reshape(-1, 1)

        # calculate spline data
        knot_x, forward_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            knot_z,
            pos_tau,
            neg_tau,
        )

        # prepare data for spline
        input_knots = knot_z.squeeze().repeat((z.shape[0], 1))
        output_knots = knot_x.squeeze().repeat((z.shape[0], 1))
        derivatives = forward_lad.squeeze().repeat((z.shape[0], 1)).exp()

        spline_x, spline_lad = univariate_forward_rqs(
            z.clamp(min=knot_z[0, 0] + 1e-6, max=knot_z[1, 0] - 1e-6),
            input_knots,
            output_knots,
            derivatives,
        )

        ex_x, ex_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            z, pos_tau, neg_tau
        )

        within = torch.logical_and(
            z < knot_z[1],
            z > knot_z[0],
        )
        x = torch.where(within, spline_x, ex_x)
        lad = torch.where(within, spline_lad, ex_lad)

        return x, lad

    @staticmethod
    def univariate_smooth_ex_inverse(x, pos_tau, neg_tau):
        knot_z = torch.tensor([-1.6, 1.6]).reshape(-1, 1)

        # calculate spline data
        knot_x, forward_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            knot_z,
            pos_tau,
            neg_tau,
        )

        # prepare data for spline
        input_knots = knot_z.squeeze().repeat((x.shape[0], 1))
        output_knots = knot_x.squeeze().repeat((x.shape[0], 1))
        derivatives = forward_lad.squeeze().repeat((x.shape[0], 1)).exp()

        spline_z, spline_lad = univariate_inverse_rqs(
            x.clamp(min=output_knots[:, 0] + 1e-6, max=output_knots[:, 1] - 1e-6),
            input_knots,
            output_knots,
            derivatives,
        )

        # inverse is negative tail param
        ex_z, ex_lad = SmoothTailSwitchMarginalTransform._real_transformation(
            x, -pos_tau, -neg_tau
        )

        within = torch.logical_and(
            x < output_knots[:, 1],
            x > output_knots[:, 0],
        )
        z = torch.where(within, spline_z, ex_z)
        lad = torch.where(within, spline_lad, ex_lad)

        return z, lad


class InterpMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(InterpMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample(torch.Size([features]))

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self.pos_tail = torch.nn.parameter.Parameter(pos_tail_init)
        self.neg_tail = torch.nn.parameter.Parameter(neg_tail_init)
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init - 1e-3))

    def inverse(self, z, context=None):
        sign = torch.sign(z)
        tail_param = torch.where(
            z > 0,
            self.pos_tail,
            self.neg_tail,
        )
        x, lad = InterpMarginalTransform._transformation(torch.abs(z), tail_param)
        lad += torch.log(self.scale)
        x = sign * x * self.scale + self.shift
        return x, lad.sum(dim=-1)

    def forward(self, x, context=None):
        # affine
        x = (x - self.shift) / self.scale

        sign = torch.sign(x)
        tail_param = torch.where(x > 0, self.pos_tail, self.neg_tail)

        z, lad = InterpMarginalTransform._transformation(
            torch.abs(x), tail_param, inverse=True
        )
        lad -= torch.log(self.scale)
        return sign * z, lad.sum(dim=-1)

    @staticmethod
    def interpolated_transformation(z, tail_param):
        # interpolated
        intermediate_tail_p = (0.5 * (tail_param + 1)).abs()  # always in [0, 1]
        _y = torch.exp(-z.new_tensor(SQRT_2 / SQRT_PI) * z)
        stable_y = _y > MIN_ERFC_INV

        erfcinv_y = torch.zeros_like(_y)
        erfcinv_y[stable_y] = _erfcinv(_y[stable_y])
        erfcinv_y[~stable_y] = _small_erfcinv(-z[~stable_y])

        low = SQRT_2 * erfcinv_y

        dlow_dx = torch.exp(erfcinv_y.square() - z.new_tensor(SQRT_2 / SQRT_PI) * z)

        high = torch.where(
            torch.special.erfc(z / SQRT_2) > MIN_ERFC_INV,
            -torch.special.erfc(z / SQRT_2).log(),
            z.log() + 0.5 * z.square() + torch.log(z.new_tensor(SQRT_PI / SQRT_2)),
        ) * z.new_tensor(SQRT_PI / SQRT_2)
        dhigh_dx = torch.where(
            z < 10,
            torch.exp(-0.5 * z.square()) / torch.special.erfc(z / SQRT_2),
            (SQRT_PI / SQRT_2) * z,
        )

        intermediate_x = intermediate_tail_p * high + (1 - intermediate_tail_p) * low

        # this could be unstable
        intermediate_lad = intermediate_tail_p * dhigh_dx
        intermediate_lad += (1 - intermediate_tail_p) * dlow_dx
        intermediate_lad = intermediate_lad.log()

        return intermediate_x, intermediate_lad

    @staticmethod
    def bisection_search(func, z_0_low, z_0_high, target, tol=1e-6, max_iter=100):
        for i in range(max_iter):
            z_mid = 0.5 * (z_0_low + z_0_high)
            f_mid = func(z_mid)
            error = target - f_mid
            if (error.abs() < tol).all():
                return z_mid

            z_0_low = torch.where(error < 0, z_0_low, z_mid)
            z_0_high = torch.where(error > 0, z_0_high, z_mid)

        return 0.5 * (z_0_low + z_0_high)

    @staticmethod
    def interpolated_inverse(x, tail_param):
        interpolated_transformation = (
            InterpMarginalTransform.interpolated_transformation
        )
        # interpolated
        low, _ = interpolated_transformation(x, torch.ones_like(x) * -1)
        high, _ = interpolated_transformation(x, torch.ones_like(x))

        z_0_low = torch.minimum(low, high)
        z_0_high = torch.maximum(low, high)

        def forward(z):
            return interpolated_transformation(z, tail_param)[0]

        z = InterpMarginalTransform.bisection_search(forward, z_0_low, z_0_high, x)
        _, inv_lad = interpolated_transformation(z.detach(), tail_param)
        lad = -inv_lad
        return z, lad

    @staticmethod
    def _transformation(z, tail_param, inverse=False):
        interpolated_inverse = InterpMarginalTransform.interpolated_inverse
        interpolated_transformation = (
            InterpMarginalTransform.interpolated_transformation
        )

        if inverse:
            tail_param = -tail_param

        if tail_param.shape[0] == 1:  # fixed parameter for each observation
            tail_param = tail_param.repeat((z.shape[0], 1))

        assert (
            z.shape[1] == tail_param.shape[1]
        ), f"Tail parameter must be 2D [1, {z.shape[1]}] or {z.shape[1]}"

        heavy_tails = tail_param > 1
        light_tails = tail_param < -1

        # fatten the tails
        heavy_tail_p = (tail_param - 1).abs()  # lambda - 1 should always be positive
        g = torch.special.erfc(z / SQRT_2)
        heavy_x = (torch.pow(g, -heavy_tail_p) - 1) / heavy_tail_p
        heavy_x *= (SQRT_PI / SQRT_2)

        heavy_lad = torch.log(g) * (-heavy_tail_p - 1)
        heavy_lad -= 0.5 * torch.square(z)

        # lighten the tails
        light_tail_p = (
            torch.abs(tail_param) - 1
        ).abs()  # |lambda| - 1 should always be positive
        inner = 1 + (SQRT_2 / SQRT_PI) * light_tail_p * z
        g = torch.pow(inner, -1 / light_tail_p)
        stable_g = g > MIN_ERFC_INV

        erfcinv_val = torch.zeros_like(z)
        erfcinv_val[stable_g] = _erfcinv(g[stable_g])
        log_g = -torch.log(inner[~stable_g]) / tail_param[~stable_g]
        erfcinv_val[~stable_g] = _small_erfcinv(log_g)
        light_x = SQRT_2 * erfcinv_val

        light_lad = (-1 - 1 / light_tail_p) * torch.log(inner)
        light_lad += torch.square(erfcinv_val)

        if inverse:
            intermediate_x, intermediate_lad = interpolated_inverse(z, -tail_param)
        else:
            intermediate_x, intermediate_lad = interpolated_transformation(
                z, tail_param
            )

        # implicitly, where not heavy or light tailed it is intermediate
        x = torch.where(heavy_tails, heavy_x, intermediate_x)
        x = torch.where(light_tails, light_x, x)

        lad = torch.where(heavy_tails, heavy_lad, intermediate_lad)
        lad = torch.where(light_tails, light_lad, lad)

        return x, lad

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False


class MaskedTailSwitchAffineTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        context_features=None,
        nn_kwargs={},
    ):
        self.features = features

        nn_kwargs = configure_nn(nn_kwargs)
        made = made_module.MADE(
            features=features,
            context_features=context_features,
            output_multiplier=self._output_dim_multiplier(),
            **nn_kwargs,
        )
        super(MaskedTailSwitchAffineTransform, self).__init__(autoregressive_net=made)

    def _output_dim_multiplier(self):
        return 4

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )

        pos_tail = 1.5 * sigmoid(unc_pos_tail) - 1.0
        neg_tail = 1.5 * sigmoid(unc_neg_tail) - 1.0
        shift = shift_param
        scale = softplus(unc_scale)

        x, lad = _tail_switch_transform(z, pos_tail, neg_tail, shift, scale)

        return x, lad.sum(dim=-1)

    def _elementwise_inverse(self, x, autoregressive_params):
        raise NotImplementedError

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
            autoregressive_params[..., 2],
            autoregressive_params[..., 3],
        )


class MaskedAutoregressiveTailAffineMarginalTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        context_features=None,
        nn_kwargs={},
    ):
        self.features = features

        nn_kwargs = configure_nn(nn_kwargs)
        made = made_module.MADE(
            features=features,
            context_features=context_features,
            output_multiplier=self._output_dim_multiplier(),
            **nn_kwargs,
        )
        super(MaskedAutoregressiveTailAffineMarginalTransform, self).__init__(
            autoregressive_net=made
        )

    def _output_dim_multiplier(self):
        return 4

    def _elementwise_forward(self, z, autoregressive_params):
        """light -> heavy"""
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )

        pos_tail = softplus(unc_pos_tail)
        neg_tail = softplus(unc_neg_tail)
        shift = shift_param
        scale = softplus(unc_scale)

        x, lad = _tail_affine_transform(z, pos_tail, neg_tail, shift, scale)
        return x, lad.sum(dim=-1)

    def _elementwise_inverse(self, x, autoregressive_params):
        """heavy -> light"""
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        pos_tail = softplus(unc_pos_tail)
        neg_tail = softplus(unc_neg_tail)
        shift = shift_param
        scale = softplus(unc_scale)

        z, lad = _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale)
        return z, lad.sum(dim=-1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
            autoregressive_params[..., 2],
            autoregressive_params[..., 3],
        )


class TailScaleShiftMarginalTransform(Transform):
    """
    A two tail scale version of the tail and scale transform.
    """

    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        pos_scale_init=None,
        neg_scale_init=None,
    ):
        self.features = features
        super(TailScaleShiftMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if shift_init is None:
            shift_init = torch.zeros([features])

        if pos_scale_init is None:
            pos_scale_init = torch.ones([features])

        if neg_scale_init is None:
            neg_scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == pos_scale_init.shape
        assert torch.Size([features]) == neg_scale_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_pos_scale = torch.nn.parameter.Parameter(inv_sftplus(pos_scale_init))
        self._unc_neg_scale = torch.nn.parameter.Parameter(inv_sftplus(neg_scale_init))

    def forward(self, x, context=None):
        pos_tail = softplus(self._unc_pos_tail)
        neg_tail = softplus(self._unc_neg_tail)
        shift = self._unc_shift
        pos_scale = 1e-5 + softplus(self._unc_pos_scale)
        neg_scale = 1e-5 + softplus(self._unc_neg_scale)

        scale_z, scale_lad = two_scale_affine_inverse(x, shift, neg_scale, pos_scale)
        z, tail_lad = _tail_inverse(scale_z, pos_tail, neg_tail)

        return z, tail_lad + scale_lad.sum(dim=-1)

    def inverse(self, z, context=None):
        """light -> heavy"""
        pos_tail = softplus(self._unc_pos_tail)
        neg_tail = softplus(self._unc_neg_tail)
        shift = self._unc_shift
        pos_scale = 1e-5 + softplus(self._unc_pos_scale)
        neg_scale = 1e-5 + softplus(self._unc_neg_scale)

        tail_x, tail_lad = _tail_forward(z, pos_tail, neg_tail)
        x, scale_lad = two_scale_affine_forward(tail_x, shift, neg_scale, pos_scale)

        return x, tail_lad + scale_lad.sum(dim=-1)


class CopulaMarginalTransform(Transform):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
    ):
        self.features = features
        super(CopulaMarginalTransform, self).__init__()
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))

    def forward(self, u, context=None):
        """light -> heavy"""
        tail_param = torch.where(
            u > 0, softplus(self._unc_pos_tail), softplus(self._unc_neg_tail)
        )
        sign = torch.sign(u)
        x, lad = _copula_transform_and_lad(torch.abs(u), tail_param)
        return sign * x, lad.sum(dim=-1)

    def inverse(self, x, context=None):
        """heavy -> light"""
        tail_param = torch.where(
            x > 0, softplus(self._unc_pos_tail), softplus(self._unc_neg_tail)
        )
        sign = torch.sign(x)
        u, lad = _copula_inverse_and_lad(torch.abs(x), tail_param)
        return sign * u, lad.sum(dim=-1)


class GTransform(Transform):
    """Alternative Gumbel transform producing exponential tails."""

    def __init__(self, features, p=1.0, kappa=1.0, epsilon=1.0, delta=1.0):
        self.features = features
        super().__init__()
        self.p = torch.tensor(p)
        self._unc_kappa = torch.nn.Parameter(inv_sftplus(torch.tensor(kappa)))
        self._unc_epsilon = torch.nn.Parameter(inv_sftplus(torch.tensor(epsilon)))
        self._unc_delta = torch.nn.Parameter(inv_sftplus(torch.tensor(delta)))

    @property
    def kappa(self):
        return softplus(self._unc_kappa)

    @property
    def epsilon(self):
        return softplus(self._unc_epsilon)

    @property
    def delta(self):
        return softplus(self._unc_delta)

    def forward(self, z, context=None):
        x, lad = _g_transform_and_lad(z, self.p, self.kappa, self.epsilon, self.delta)
        return x, lad.sum(dim=-1)

    def inverse(self, x, context=None):
        z, lad = _g_inverse_and_lad(x, self.p, self.kappa, self.epsilon, self.delta)
        return z, lad.sum(dim=-1)


class ExpTailTransform(Transform):
    """Transform exponential tail to Frechet tail."""

    def __init__(self, features, lam=1.0):
        self.features = features
        super().__init__()
        self._unc_lam = torch.nn.Parameter(inv_sftplus(torch.tensor(lam)))

    @property
    def lam(self):
        return softplus(self._unc_lam)

    def forward(self, x, context=None):
        y, lad = _exp_tail_transform_and_lad(x, self.lam)
        return y, lad.sum(dim=-1)

    def inverse(self, y, context=None):
        x, lad = _exp_tail_inverse_and_lad(y, self.lam)
        return x, lad.sum(dim=-1)


class FullTailMarginalTransform(Transform):
    """Composite transform applying G then exponential Frechet T."""

    def __init__(self, features, *, p=1.0, kappa=1.0, epsilon=1.0, delta=1.0, lam=1.0):
        super().__init__()
        self.features = features
        self.g = GTransform(features, p=p, kappa=kappa, epsilon=epsilon, delta=delta)
        self.t = ExpTailTransform(features, lam=lam)

    def forward(self, z, context=None):
        x, lad1 = self.g.forward(z, context)
        y, lad2 = self.t.forward(x, context)
        return y, lad1 + lad2

    def inverse(self, y, context=None):
        x, lad2 = self.t.inverse(y, context)
        z, lad1 = self.g.inverse(x, context)
        return z, lad1 + lad2


class MaskedExtremeAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        nn_kwargs,
        context_features=None,
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            context_features=context_features,
            output_multiplier=self._output_dim_multiplier(),
            **nn_kwargs,
        )
        # init at low value
        made.final_layer.bias = torch.nn.Parameter(
            -2 * torch.ones_like(made.final_layer.bias)
        )
        super(MaskedExtremeAutoregressiveTransform, self).__init__(made)

    def _output_dim_multiplier(self):
        return 4

    def _elementwise_forward(self, z, autoregressive_params):
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        shift = shift_param
        pos_tail = softplus(unc_pos_tail)  # (0, inf)
        neg_tail = softplus(unc_neg_tail)  # (0, inf)
        scale = softplus(unc_scale)

        x, lad = _tail_affine_transform(z, pos_tail, neg_tail, shift, scale)
        return x, lad.sum(dim=-1)

    def _elementwise_inverse(self, x, autoregressive_params):
        unc_pos_tail, unc_neg_tail, unc_scale, shift_param = self._unconstrained_params(
            autoregressive_params
        )
        shift = shift_param
        pos_tail = softplus(unc_pos_tail)  # (0, inf)
        neg_tail = softplus(unc_neg_tail)  # (0, inf)
        scale = softplus(unc_scale)

        z, lad = _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale)
        return z, lad.sum(dim=-1)

    def _unconstrained_params(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return (
            autoregressive_params[..., 0],
            autoregressive_params[..., 1],
            autoregressive_params[..., 2],
            autoregressive_params[..., 3],
        )


class Marginal(Transform):
    def __init__(self, marginal_transforms):
        self.marginal_transforms = marginal_transforms
        super(Marginal, self).__init__()

    def inverse(self, z, context=None):
        xs = []
        lad = 0.0
        for dim_ix, mt in enumerate(self.marginal_transforms):
            x, _lad = mt.inverse(z[:, [dim_ix]], context)
            xs.append(x)
            lad += _lad
        return torch.hstack(xs), lad

    def forward(self, z, context=None):
        xs = []
        lad = 0.0
        for dim_ix, mt in enumerate(self.marginal_transforms):
            x, _lad = mt.forward(z[:, [dim_ix]], context)
            xs.append(x)
            lad += _lad
        return torch.hstack(xs), lad


def bisection_search(func, x_0_low, x_0_high, target, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        x_mid = 0.5 * (x_0_low + x_0_high)
        f_mid = func(x_mid)
        error = f_mid - target
        if (error.abs() < tol).all():
            return x_mid

        x_0_low = torch.where(error < 0, x_mid, x_0_low)
        x_0_high = torch.where(error > 0, x_mid, x_0_high)

    return 0.5 * (x_0_low + x_0_high)



# The implementation here makes no sense
# class Mixture(Transform):
#     def __init__(self, transform_1, transform_2):
#         self.transform_1 = transform_1
#         self.transform_2 = transform_2
#         self._unc_mix_param = torch.nn.Parameter(torch.tensor(0.0))
#         super(Transform, self).__init__()

#     def inverse(self, x, context=None):
#         z_1, lad_1 = self.transform_1.inverse(x, context)
#         z_2, lad_2 = self.transform_2.inverse(x, context)

#         mix_param = softplus(self._unc_mix_param)

#         z = z_1 * mix_param + z_2 * (1 - mix_param)
#         z = lad_1 * mix_param + lad_2 * (1 - mix_param)

#         return torch.hstack(xs), lad

#     def forward(self, z, context=None):
#         x_1, _ = self.transform_1.forward(z)
#         x_2, _ = self.transform_2.forward(z)
#         x_1_higher = x_1 > x_2
#         x_0_high = torch.where(x_1_higher, x_1, x_2)
#         x_0_low = torch.where(x_1_higher, x_2, x_1)

#         inverse_trans = lambda x: self.inverse(x, context)
#         x = bisection_search(
#             inverse_trans, x_0_low, x_0_high, target=z, tol=1e-6, max_iter=100
#         )  # which x gives z?
#         _, inv_lad = inverse_trans(x.detach())
#         lad = -inv_lad
#         return x, lad

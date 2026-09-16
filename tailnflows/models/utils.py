import torch
import numpy as np
from nflows.transforms.base import Transform
from torch.nn.functional import softplus
from nflows.utils import torchutils


def inv_sftplus(x):
    return x + torch.log(-torch.expm1(-x))


def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)


class Marginal(Transform):
    def __init__(self, dim, transform):
        super().__init__()
        self.transform = transform
        self.params = [
            torch.nn.Parameter(
                torch.zeros(
                    dim * shape,
                )
            )
            for shape in transform.param_shape
        ]
        for ix, p in enumerate(self.params):
            self.register_parameter(f"p_{ix}", p)

    def forward(self, z, context=None):
        tiled_p = tuple(torch.tile(p, (z.shape[0], 1)) for p in self.params)
        x, logabsdet = self.transform.forward_and_lad(z, *tiled_p)
        return x, logabsdet

    def inverse(self, z, context=None):
        tiled_p = tuple(torch.tile(p, (z.shape[0], 1)) for p in self.params)
        z, lad = self.transform.inverse_and_lad(z, *tiled_p)
        return z, lad


class Softplus(Transform):
    def __init__(self, temperature=1.0, learn_temperature=False):
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    # data -> noise
    def forward(self, outputs, context=None):
        if torch.min(outputs) < 0:
            raise InputOutsideDomain()

        inputs = inv_sftplus(outputs)
        logabsdet = -torchutils.sum_except_batch(
            torch.where(
                inputs < 0,
                self.temperature * inputs
                - torch.log1p(torch.exp(self.temperature * inputs)),
                -torch.log1p(torch.exp(-self.temperature * inputs)),
            )
        )
        return inputs, logabsdet

    # noise -> data
    def inverse(self, inputs, context=None):
        outputs = softplus(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.where(
                inputs < 0,
                self.temperature * inputs
                - torch.log1p(torch.exp(self.temperature * inputs)),
                -torch.log1p(torch.exp(-self.temperature * inputs)),
            )
        )
        return outputs, logabsdet


def invertibility_check(flow, samps: torch.Tensor, tol: float = 1e-4) -> bool:
    """ Check if a flow model is invertible by sending a batch of points through the flow (in both directions).

    Args:
        flow: The flow model to test invertibility for. Must implement forward() and inverse() methods.
        samps (torch.Tensor): A data batch (shape [batch, features]), e.g. sampled from N(0,I).
        tol (float): Numerical tolerance for the comparison (default: 1e-4).

    Returns:
        bool: True if the inverse->forward processed batch agrees with the original batch (up to the numerical tolerance), oterwise False.
    """
    is_invertible = True

    # Apply inverse (noise->data) transformation
    processed_samps = flow._transform.inverse(samps)
    if isinstance(processed_samps, tuple):
        processed_samps = processed_samps[0] # only samples, not lad

    # Apply forward(data->noise) transformation
    processed_samps = flow._transform.forward(processed_samps)
    if isinstance(processed_samps, tuple):
        processed_samps = processed_samps[0] # only samples, not lad

    # Inspect difference between original and processed samples
    if not torch.allclose(processed_samps, samps, rtol=tol, atol=tol):
        is_invertible = False

    return is_invertible
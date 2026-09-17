
import torch
import torch.nn as nn
import normflows as nf
import numpy as np
import math
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from tqdm import tqdm

from typing import Optional, Union, Sequence

import openturns as ot

from tailnflows.models.extreme_transformations import TailAffineMarginalTransform, ModifiedTailAffineMarginalTransform, SoftLogMarginalTransform
from nflows.transforms import Transform

# Sinusoidal time embedding
class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed 1-d time into {time_emb_dim}-dimensional space.

        Args:
            t (torch.Tensor): Shape [batch] or [batch, 1].

        Returns:
            temb (torch.Tensor): Shape [batch, time_emb_dim].
        """
        if t.dim() == 1:
            t = t[:, None]

        half_dim = self.time_emb_dim // 2
        freqs = torch.exp(
            torch.linspace(
                0,
                torch.log(torch.tensor(1000.0, device=t.device)),
                half_dim,
                device=t.device,
            )
        )
        args = t * freqs[None, :] * 2.0 * torch.pi
        temb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.time_emb_dim % 2 == 1:
            temb = torch.cat([temb, t], dim=-1)

        return temb


# Residual layer with smooth SiLU activation
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape [batch, dim]

        Returns:
            torch.Tensor: Shape [batch, dim]
        """
        return x + self.net(x)


# Time-dependent vector field for Flow Matching models
class FMVectorField(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        time_emb_dim: int = 16,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.time_embedding = TimeEmbedding(time_emb_dim)

        input_dim = x_dim + time_emb_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, x_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape [batch, x_dim].
            t (torch.Tensor): Shape [batch] or [batch, 1].

        Returns:
            v (torch.Tensor): Velocity vector field at point x and time t (shape [batch, x_dim]).
        """

        # Ensure t has the same batch size as x
        if t.dim() == 0:  # scalar
            t = t.expand(x.shape[0])
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])
        
        temb = self.time_embedding(t)
        h = torch.cat([x, temb], dim=-1)

        h = self.input_layer(h)

        for block in self.blocks:
            h = block(h)

        v = self.output_layer(h)
        return v


# Conditional FM loss for linear Gaussian prob. paths
mse_loss = nn.MSELoss() #reduction="none") # use non for weighted loss..
def cfm_loss(vf: FMVectorField, x1: torch.Tensor, q0: nf.distributions.BaseDistribution) -> torch.Tensor:
    """
    Conditional Flow Matching loss for linear (OT) Gaussian Probability Path.

    Args:
        vf (FMVectorField): FM vector field.
        x1 (torch.Tensor): Data batch (shape: [batch_size, features]).
        q0 (nf.distributions.BaseDistribution): Base distribution of the FM model (Usually N(0,I)).

    Returns:
        loss (torch.Tensor): loss on data batch (shape: []).
    """
    batch_size = x1.shape[0]
    device = x1.device

    # Instantiate OT probability path
    path = AffineProbPath(scheduler=CondOTScheduler())

    # sample from base distribution
    x0, _ = q0.forward(batch_size)
    x0 = x0.to(device=device)

    # sample time uniformly
    t = torch.rand(batch_size, device=device).to(device)

    # Sample probability path
    path_sample = path.sample(x0, x1, t)

    # Conditional FM l2 loss
    loss = mse_loss(vf(path_sample.x_t.to(device), path_sample.t.to(device)), path_sample.dx_t.to(device))
    loss = loss.mean()

    return loss

# Final Layers
class IdentityTransform(Transform):
    """ Identity function in feature space. """

    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, z: torch.Tensor, context=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z (torch.Tensor): Shape [batch, features].

        Returns:
            z, lad (tuple[torch.Tensor, torch.Tensor]): lad (=0) is summed over dimension: Shape [batch].
        """
        lad = torch.zeros_like(z)
        return z, lad.sum(dim=-1)

    def inverse(self, x: torch.Tensor, context=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Shape [batch, features].

        Returns:
            x, lad (tuple[torch.Tensor, torch.Tensor]): lad (=0) is summed over dimension: Shape [batch].
        """
        lad = torch.zeros_like(x)
        return x, lad.sum(dim=-1)
    

class FM_TTF_model(torch.nn.Module):
    """ Implementation of a Flow Matching model together with an optional final tail (TTF) transformation. """

    def __init__(self, device: torch.device, vf: FMVectorField, q0: nf.distributions.BaseDistribution, tail_trafo: Optional[Transform] = None):
        """
        Build combined model consisting of a FM velocity field and a final tail transformation.

        Args:
            device (torch.device): Device.
            vf (FMVectorField): Flow Matching velocity field.
            q0 (nf.distributions.BaseDistribution): Base distribution of the FM model.
            tail_trafo (Transform, optional): Final tail transformation. If given None, the identity transformation is used. Defaults to None.
        """
        super().__init__()
        self.features = vf.x_dim
        self.device = device
        self.vf = vf.to(device)
        self.q0 = q0.to(device)
        if tail_trafo is None:
            tail_trafo = IdentityTransform(features=vf.x_dim) # Default: Identity Transformation
        self.tail_trafo = tail_trafo.to(device)

        assert self.features == self.tail_trafo.features, f"Mismatch in dimension of vf ({self.features}) and dimension of tail_trafo ({self.tail_trafo.features})."

        # Fix TTF params
        if isinstance(self.tail_trafo, (TailAffineMarginalTransform, ModifiedTailAffineMarginalTransform)):
            print("Currently, learning the TTF params is not possible for FM-TTF models. Therefore, TTF params will be fixed.")
            if isinstance(self.tail_trafo, ModifiedTailAffineMarginalTransform):
                self.tail_trafo.fix_all()
            elif isinstance(self.tail_trafo, TailAffineMarginalTransform):
                self.tail_trafo.fix_tails()
                print("Fixed tailparams.")
                self.tail_trafo.shift.requires_grad = False
                print("Fixed location params.")
                self.tail_trafo._unc_scale.requires_grad = False
                print("Fixed scale params.")

    @torch.no_grad()
    def sample(self, num_samps: int, method: str = "dopri5", step_size: Optional[float] = None, **ode_extras) -> torch.Tensor:
        """ Sample from model.

        Args:
            num_samps (int): Number of samples.
            method (str, optional): Method for ODE solver. Defaults to "dopri5".
            step_size (float, optional): Stepsize for ODE solver. Must be None for adaptive solvers. Defaults to None.
            **ode_extras: Additional config for ODE solver.

        Returns:
            x1 (torch.Tensor): Samples (shape: [num_samps, self.features])
        """
        # Sample from base distribution
        x0 = self.q0.sample(num_samps)
        if isinstance(x0, tuple):
            x0 = x0[0] # take only samples, not log_prob
        x0 = x0.to(self.device)

        # Solve forward ODE
        solver = ODESolver(ModelWrapper(self.vf))
        y = solver.sample(x0, step_size=step_size, method=method, **ode_extras)

        # Apply tail transformation
        x1, _ = self.tail_trafo.forward(y)

        return x1

    @torch.no_grad()
    def log_prob(self, x1: torch.Tensor, method: str = "dopri5", exact_divergence: bool = False, step_size: Optional[float] = None, **ode_extras) -> torch.Tensor:
        """ Compute log_prob on data batch x1.

        Args:
            x1 (torch.Tensor): Data batch (shape: [batch_size, self.features])
            method (str, optional): Method for ODE solver. Defaults to "dopri5".
            exact_divergence (bool, optional): Whether to use exact diversion or Hutchinson estimator in ODE solver. Defaults to False.
            step_size (float, optional): Stepsize for ODE solver. Must be None for adaptive solvers. Defaults to None.
            **ode_extras: Additional config for ODE solver.

        Returns:
            log_p (torch.Tensor): log prob values for all data points (summed over feature dimension) (shape: [batch]).
        """
        x1 = x1.to(self.device)

        print(f"Computing likelihood on tensor of shape {x1.shape}...")

        # Apply inverse tail transformation
        y, lad = self.tail_trafo.inverse(x1)
        y = y.to(self.device)
        lad = lad.to(self.device)

        # Solve reverse-time ODE
        solver = ODESolver(ModelWrapper(self.vf))
        _ , log_p = solver.compute_likelihood(y, self.q0.log_prob, step_size=step_size, method=method, exact_divergence=exact_divergence, **ode_extras)
        log_p = log_p.to(self.device)

        # Change of variables formula
        log_p = log_p + lad

        return log_p

    @torch.no_grad()
    def get_average_nll(self, samples: torch.Tensor, NLL_per_dim: bool = False, method: str = 'dopri5', batch_size: int = 64000, exact_divergence: bool = True, step_size: Optional[float] = None, **ode_extras) -> float:
        """ Compute average NLL over target samples. Computation is split in batches to reduce peak usage.

        Args:
            samples (torch.Tensor): Samples from the target distribution (shape: [num_samps, self.features]).
            method (str, optional): Method for ODE sovler. Defaults to 'dopri5'.
            batch_size (int, optional): Size of batches on which likelihood is computed simultaneously. Defaults to 4096.
            exact_divergence (bool, optional): Whether to use exact divergence or Hutchinson estimator in ODE solver. Defaults to True.
            step_size (float, optional): Stepsize for ODE solver. Must be None for adaptive solvers. Defaults to None.
            **ode_extras: Additional config for ODE solver.

        Returns:
            nll (float): Average NLL per sample. If NLL_per_dim is True, returns average NLL per sample per dimension.
        """
        # Sample independent target data points
        x1 = samples.to(device=self.device)

        # Compute log_p in batches
        total_log_p = 0.0

        for start in tqdm(range(0, samples.shape[0], batch_size)):

            # Compute batch log_p
            end = min(start + batch_size, samples.shape[0])
            batch = x1[start:end]
            log_p = self.log_prob(batch, method=method, exact_divergence=exact_divergence, step_size=step_size, **ode_extras)

            # Add to total log_p and delete batch results to free memory
            total_log_p += log_p.sum().item()
            del batch, log_p
            # if device.type == 'cuda':
            #     torch.cuda.empty_cache()

        # Average total NLL over samples
        nll = -total_log_p / samples.shape[0]
        if NLL_per_dim:
            nll = nll / self.features

        return nll

    def fit(self, samples: torch.Tensor, num_epochs: int = 100, batch_size: int = 1024, show_every: Optional[int] = None, lr: float = 1e-4, weight_decay: float = 1e-5, model_name: Optional[str] = None):
        """ Train the model on target distribution. Trains only the FM vector field, not the final transformation.

        Args:
            samples (torch.Tensor): Samples from the target distribution (shape: [num_samps, self.features]).
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Defaults to 512.
            show_every (int, optional): Period for visualizing intermediate results (only for 2dim). If None, no intermediate results are computed. Defaults to None.
            lr (float, optional): Learning rate for optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-5.
            model_name (str, optional): Name of the model for saving intermediate results. Defaults to None.
        """
        print("Starting training with samples shape:", samples.shape)
        samples = samples.to(device=self.device)

        # create data loader
        dataset = torch.utils.data.TensorDataset(samples)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        first_batch = next(iter(data_loader))

        # Prepare loss history
        loss_hist = np.array([])

        # Apply inverse tail transform
        y, _ = self.tail_trafo.inverse(first_batch[0].to(device=self.device))
        # Compute and log initial loss
        loss = cfm_loss(self.vf, y.to(self.device), self.q0)
        loss_hist = np.append(loss_hist, loss.to('cpu').item())

        # Training loop
        optimizer = torch.optim.AdamW(self.vf.parameters(), lr=lr, weight_decay=weight_decay)

        pbar = tqdm(range(num_epochs), desc="Epochs")
        for epoch in pbar:
            for batch in data_loader:
                optimizer.zero_grad()

                # Get training batch
                x = batch[0].to(device=self.device)
                # Apply inverse tail transform
                y, _ = self.tail_trafo.inverse(x)

                # Compute loss
                loss = cfm_loss(self.vf, y, self.q0)

                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()

                    # clip grad norm to stabilize training... 
                    torch.nn.utils.clip_grad_norm_(
                        self.vf.parameters(),
                        max_norm=5.0,
                    )
                    optimizer.step()

                # log loss in progress bar
                loss_hist = np.append(loss_hist, loss.to('cpu').item())
                pbar.set_postfix({"running loss": loss_hist[-100:].mean()})

        return loss_hist
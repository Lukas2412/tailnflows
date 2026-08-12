# =====================================================================
# = inspired by : https://github.com/VincentStimper/normalizing-flows =
# =====================================================================
from typing import Optional
import torch
from torch import nn
import numpy as np
import openturns as ot

# Seed ot with random value
ot.RandomGenerator.SetSeed(np.random.randint(0, 100000))

class Target(nn.Module):
    """
    Sample target distributions to test models.
    """

    def __init__(self):
        
        super().__init__()

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        
        raise NotImplementedError("The log probability is not implemented yet.")

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        
        raise NotImplementedError("The sampling is not implemented yet.")
    

class NealsFunnel(Target):
        """
        Bivariate Neal's funnel distribution with location parameter gamma on z1.
        z1 is light-tailed ~ N(gamma,1)
        z2 is heavy-tailed ~ N(0, exp(z1))
        """

        def __init__(self, gamma: float = 0.0):
            """Constructor.

            Args:
              gamma (float): Location parameter for z1
            """
            super().__init__()
            self.gamma = gamma

        def __str__(self):
            return f"NealsFunnel(gamma={self.gamma})"

        def log_prob(self, z: torch.Tensor) -> torch.Tensor:
            """
            Calculate log probability of a batch of samples.

            Args:
              z (torch.Tensor): Batch of random variables to determine log probability for (shape: [batch_size, 2])

            Returns:
              log_prob (torch.Tensor): log probability for each batch element (shape: [batch_size])
            """
            z1 = z[:, 0]
            z2 = z[:, 1]
            log_pz1 = - 0.5 * torch.log(2 * torch.tensor(torch.pi)) - 0.5 * (z1 - self.gamma) ** 2
            log_pz2 = - 0.5 * torch.log(2 * torch.tensor(torch.pi)) - 0.5 * z1 - 0.5 * (z2 ** 2) * torch.exp(-z1)
            return log_pz1 + log_pz2

        def sample(self, num_samples: int = 1) -> torch.Tensor:
            """Sample from Neal's funnel distribution.

            Args:
              num_samples (int): Number of samples to draw

            Returns:
              z (torch.Tensor): Samples drawn from the distribution (shape: [num_samples, 2])
            """
            z1 = self.gamma + torch.randn(num_samples, 1)  # N(gamma,1)
            z2 = torch.exp(0.5 * z1) * torch.randn(num_samples, 1)  # N(0, exp(z1))
            return torch.cat([z1, z2], dim=1)
        

class Student(Target):
        """
        Multivariate Student's t-distribution.
        """
        def __init__(self, dim: int =2, df: float = 3.0):
            """Constructor.

            Args:
              dim (int): Dimension of the distribution
              df (float): Positive degrees of freedom parameter
            """
            super().__init__()
            self.dim = dim
            self.df = df
            self.dist = ot.Student(df, dim)

        def __str__(self):
            return f"Student(dim={self.dim}, df={self.df})"

        def log_prob(self, z: torch.Tensor) -> torch.Tensor:
            """Calculate log probability of batch of samples.

            Args:
              z (torch.Tensor): Batch of inputs to determine log probability for (shape: [batch_size, dim])

            Returns:
              log_prob (torch.Tensor): log probability for each batch element (shape: [batch_size])
            """
            z_np = z.detach().cpu().numpy()
            log_probs = self.dist.computeLogPDF(z_np)
            return torch.tensor(log_probs, dtype=z.dtype, device=z.device)

        def sample(self, num_samples: int = 1) -> torch.Tensor:
            """Sample from Student's t-distribution

            Args:
              num_samples (int): Number of samples to draw

            Returns:
              z (torch.Tensor): Samples drawn from the distribution (shape: [num_samples, dim])
            """
            samples_np = self.dist.getSample(num_samples)
            return torch.tensor(samples_np, dtype=torch.float64)
        
        
class GaussianCopula(Target):
        """
        Multivariate Gaussian Copula with mixed-tailed marginals (light-tailed Gaussian and heavy-tailed Student-t).
        """
        def __init__(self, num_light: int, num_heavy: int, df: float = 3.0, corr_matrix: Optional[ot.CorrelationMatrix] = None):
            """Constructor.

            Args:
              num_light (int): Number of light-tailed dimensions (Gaussian)
              num_heavy (int): Number of heavy-tailed dimensions (Student-t)
              df (float): Degrees of freedom for the heavy-tailed Student-t marginals
              corr_matrix (ot.CorrelationMatrix | None): Correlation matrix for the Gaussian copula (if None, identity matrix is used)
            """
            super().__init__()
            self.df = df
            self.num_light, self.num_heavy = num_light, num_heavy
            self.dim = num_light + num_heavy
            self.corr_matrix = corr_matrix if corr_matrix is not None else ot.CorrelationMatrix(self.dim)
            self.copula = ot.NormalCopula(self.corr_matrix)
            self.light_marginals = [ot.Normal(0.0, 1.0) for _ in range(num_light)]
            self.heavy_marginals = [ot.Student(df, 1) for _ in range(num_heavy)]
            self.marginals = self.light_marginals + self.heavy_marginals
            self.dist = ot.JointDistribution(self.marginals, self.copula)

        def __str__(self):
            return f"GaussianCopula(num_light={self.num_light}, num_heavy={self.num_heavy}, df={self.df})"

        def log_prob(self, z: torch.Tensor) -> torch.Tensor:
            """Calculate log probability of batch of samples

            Args:
              z (torch.Tensor): Batch of inputs to determine log probability for (shape: [batch_size, dim])

            Returns:
              log_prob (torch.Tensor): log probability for each batch element (shape: [batch_size])
            """
            z_np = z.detach().cpu().numpy()
            log_probs = self.dist.computeLogPDF(z_np)
            return torch.tensor(log_probs, dtype=z.dtype, device=z.device)

        def sample(self, num_samples: int = 1) -> torch.Tensor:
            """Sample from Gaussian Copula distribution

            Args:
              num_samples (int): Number of samples to draw
            
            Returns:
              z (torch.Tensor): Samples drawn from the distribution (shape: [num_samples, dim])
            """
            samples_np = self.dist.getSample(num_samples)
            return torch.tensor(samples_np, dtype=torch.float64)
        

class CustomGaussianCopula(Target):
     """
     Gaussian Copula Distribution with specifiable marginals.
     """
     def __init__(self, marginals: list[ot.Distribution], corr_matrix: Optional[ot.CorrelationMatrix] = None):
         """
         Constructor.

         Args:
           marginals (list[ot.Distribution]): List of univariate marginal distributions for each dimension
           corr_matrix (ot.CorrelationMatrix | None): Correlation matrix for the Gaussian copula (if None, identity matrix is used)
         """
         super().__init__()
         self.dim = len(marginals)
         self.corr_matrix = corr_matrix if corr_matrix is not None else ot.CorrelationMatrix(self.dim)
         self.copula = ot.NormalCopula(self.corr_matrix)
         self.marginals = marginals
         self.dist = ot.JointDistribution(self.marginals, self.copula)

     def __str__(self):
         return f"CustomGaussianCopula(dim={self.dim})"

     def log_prob(self, z: torch.Tensor) -> torch.Tensor:
         """
         Calculate log probability of batch of samples

         Args:
           z (torch.Tensor): Batch of inputs to determine log probability for (shape: [batch_size, dim])

         Returns:
           log_prob (torch.Tensor): log probability for each batch element (shape: [batch_size])
         """
         z_np = z.detach().cpu().numpy()
         log_probs = self.dist.computeLogPDF(z_np)
         return torch.tensor(log_probs, dtype=z.dtype, device=z.device)

     def sample(self, num_samples: int = 1) -> torch.Tensor:
         """
         Sample from Custom Gaussian Copula distribution

         Args:
           num_samples (int): Number of samples to draw
        
         Returns:
           z (torch.Tensor): Samples drawn from the distribution (shape: [num_samples, dim])
         """
         samples_np = self.dist.getSample(num_samples)
         return torch.tensor(samples_np, dtype=torch.float64)
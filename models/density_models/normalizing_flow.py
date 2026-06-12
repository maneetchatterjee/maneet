"""
Normalizing Flow for Change Distribution Modeling

Implements RealNVP-based normalizing flow for modeling p(z_Δ | y=change).
More expressive than GMM for complex distributions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
import numpy as np


class AffineCoupling(nn.Module):
    """
    Affine coupling layer for RealNVP.
    
    Splits input into two parts and applies affine transformation
    to one part conditioned on the other.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 256, mask_type: str = 'even'):
        super(AffineCoupling, self).__init__()
        
        self.dim = dim
        self.mask_type = mask_type
        
        # Create mask (alternating for each layer)
        self.register_buffer('mask', self._create_mask(dim, mask_type))
        
        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def _create_mask(self, dim: int, mask_type: str) -> torch.Tensor:
        """Create binary mask for coupling."""
        mask = torch.zeros(dim)
        if mask_type == 'even':
            mask[::2] = 1
        else:  # odd
            mask[1::2] = 1
        return mask
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation.
        
        Args:
            x: Input (B, dim)
            
        Returns:
            (transformed_x, log_det_jacobian)
        """
        masked_x = x * self.mask
        
        s = self.scale_net(masked_x)
        t = self.translate_net(masked_x)
        
        # Apply transformation to unmasked part
        y = masked_x + (1 - self.mask) * (x * torch.exp(s) + t)
        
        # Log determinant of Jacobian
        log_det = torch.sum((1 - self.mask) * s, dim=1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation.
        
        Args:
            y: Transformed input (B, dim)
            
        Returns:
            Original x
        """
        masked_y = y * self.mask
        
        s = self.scale_net(masked_y)
        t = self.translate_net(masked_y)
        
        x = masked_y + (1 - self.mask) * (y - t) * torch.exp(-s)
        
        return x


class RealNVP(nn.Module):
    """
    RealNVP normalizing flow for density estimation.
    
    Args:
        input_dim: Dimension of input embeddings
        n_flows: Number of coupling layers
        hidden_dim: Hidden dimension for coupling networks
    """
    
    def __init__(
        self,
        input_dim: int,
        n_flows: int = 6,
        hidden_dim: int = 256
    ):
        super(RealNVP, self).__init__()
        
        self.input_dim = input_dim
        self.n_flows = n_flows
        
        # Create alternating coupling layers
        self.flows = nn.ModuleList([
            AffineCoupling(
                input_dim,
                hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(n_flows)
        ])
        
        # Base distribution (standard Gaussian)
        self.register_buffer('base_mean', torch.zeros(input_dim))
        self.register_buffer('base_std', torch.ones(input_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform x to base distribution and compute log probability.
        
        Args:
            x: Input embeddings (B, input_dim)
            
        Returns:
            (z, log_prob) where z is in base space
        """
        log_det_sum = 0
        z = x
        
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
        
        # Log probability under base distribution
        log_prob_base = -0.5 * torch.sum(z ** 2, dim=1) - \
                        0.5 * self.input_dim * np.log(2 * np.pi)
        
        # Total log probability
        log_prob = log_prob_base + log_det_sum
        
        return z, log_prob
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform from base distribution to data space.
        
        Args:
            z: Samples from base distribution
            
        Returns:
            Samples in data space
        """
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the learned distribution.
        
        Args:
            n_samples: Number of samples
            device: Device to generate samples on
            
        Returns:
            Samples (n_samples, input_dim)
        """
        z = torch.randn(n_samples, self.input_dim, device=device)
        return self.inverse(z)
    
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood of data.
        
        Args:
            x: Data points (B, input_dim)
            
        Returns:
            Log-likelihoods (B,)
        """
        _, log_prob = self.forward(x)
        return log_prob


class ChangeNormalizingFlow:
    """
    Wrapper for normalizing flow-based density estimation of change embeddings.
    
    Provides sklearn-like interface matching ChangeGMM.
    
    Args:
        input_dim: Dimension of change embeddings
        n_flows: Number of coupling layers
        hidden_dim: Hidden dimension
        lr: Learning rate for training
        max_epochs: Maximum training epochs
        batch_size: Batch size for training
        device: Device for computation
    """
    
    def __init__(
        self,
        input_dim: int,
        n_flows: int = 6,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.input_dim = input_dim
        self.n_flows = n_flows
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.model = RealNVP(input_dim, n_flows, hidden_dim).to(self.device)
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray, verbose: bool = False) -> 'ChangeNormalizingFlow':
        """
        Train normalizing flow on change embeddings.
        
        Args:
            embeddings: Change embeddings (N, input_dim)
            verbose: Print training progress
            
        Returns:
            self
        """
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Convert to tensor
        data = torch.FloatTensor(embeddings).to(self.device)
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            for batch, in loader:
                optimizer.zero_grad()
                
                _, log_prob = self.model(batch)
                loss = -log_prob.mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def log_likelihood(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of embeddings.
        
        Args:
            embeddings: Change embeddings (N, input_dim)
            
        Returns:
            Log-likelihoods (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("Flow must be fitted before computing log-likelihood")
        
        if torch.is_tensor(embeddings):
            data = embeddings.to(self.device)
        else:
            data = torch.FloatTensor(embeddings).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            log_prob = self.model.log_likelihood(data)
        
        return log_prob.cpu().numpy()
    
    def score(self, embeddings: np.ndarray) -> float:
        """
        Compute average log-likelihood.
        
        Args:
            embeddings: Change embeddings
            
        Returns:
            Average log-likelihood
        """
        return np.mean(self.log_likelihood(embeddings))
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from learned distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Samples (n_samples, input_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Flow must be fitted before sampling")
        
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(n_samples, self.device)
        
        return samples.cpu().numpy()
    
    def save(self, path: str):
        """Save model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        torch.save({
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim,
            'n_flows': self.n_flows,
            'hidden_dim': self.hidden_dim,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        self.n_flows = checkpoint['n_flows']
        self.hidden_dim = checkpoint['hidden_dim']
        self.is_fitted = checkpoint['is_fitted']
        
        self.model = RealNVP(
            self.input_dim,
            self.n_flows,
            self.hidden_dim
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])


def build_normalizing_flow(
    input_dim: int,
    n_flows: int = 6,
    hidden_dim: int = 256,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ChangeNormalizingFlow:
    """
    Factory function to create ChangeNormalizingFlow.
    
    Args:
        input_dim: Embedding dimension
        n_flows: Number of flow layers
        hidden_dim: Hidden dimension
        device: Computation device
        
    Returns:
        ChangeNormalizingFlow instance
    """
    return ChangeNormalizingFlow(
        input_dim=input_dim,
        n_flows=n_flows,
        hidden_dim=hidden_dim,
        device=device
    )


if __name__ == "__main__":
    print("Testing RealNVP Normalizing Flow:")
    
    # Generate synthetic embeddings
    torch.manual_seed(42)
    train_embeddings = torch.randn(1000, 128)
    test_embeddings = torch.randn(100, 128)
    
    # Fit flow
    flow = build_normalizing_flow(input_dim=128, n_flows=4, device='cpu')
    flow.fit(train_embeddings.numpy(), verbose=True)
    print(f"\nFlow fitted: {flow.is_fitted}")
    
    # Compute log-likelihoods
    train_ll = flow.log_likelihood(train_embeddings)
    test_ll = flow.log_likelihood(test_embeddings)
    
    print(f"\nTrain log-likelihood: mean={train_ll.mean():.3f}, std={train_ll.std():.3f}")
    print(f"Test log-likelihood: mean={test_ll.mean():.3f}, std={test_ll.std():.3f}")
    
    # Sample from flow
    samples = flow.sample(10)
    print(f"\nSampled shape: {samples.shape}")

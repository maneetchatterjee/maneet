"""
Gaussian Mixture Model for Change Distribution Modeling

Models the distribution of change embeddings p(z_Δ | y=change) using GMM.
Used for detecting out-of-distribution changes via likelihood thresholding.
"""

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional, Tuple
import pickle


class ChangeGMM:
    """
    Gaussian Mixture Model for modeling in-distribution change embeddings.
    
    Trains on change embeddings from the training set and computes
    log-likelihood to detect OOD changes at test time.
    
    Args:
        n_components: Number of Gaussian components
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for EM algorithm
    """
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        random_state: int = 42,
        max_iter: int = 100
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter,
            verbose=0
        )
        
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'ChangeGMM':
        """
        Fit GMM to change embeddings.
        
        Args:
            embeddings: Change embeddings z_Δ from training change samples
                       Shape: (N, embedding_dim)
        
        Returns:
            self
        """
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        
        self.gmm.fit(embeddings)
        self.is_fitted = True
        
        return self
    
    def log_likelihood(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of embeddings under the fitted distribution.
        
        Args:
            embeddings: Change embeddings to evaluate
                       Shape: (N, embedding_dim)
        
        Returns:
            Log-likelihood scores, shape (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("GMM must be fitted before computing log-likelihood")
        
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        
        return self.gmm.score_samples(embeddings)
    
    def score(self, embeddings: np.ndarray) -> float:
        """
        Compute average log-likelihood (for model evaluation).
        
        Args:
            embeddings: Change embeddings
            
        Returns:
            Average log-likelihood
        """
        return np.mean(self.log_likelihood(embeddings))
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict GMM component assignment for each embedding.
        
        Args:
            embeddings: Change embeddings
            
        Returns:
            Component labels, shape (N,)
        """
        if not self.is_fitted:
            raise RuntimeError("GMM must be fitted before prediction")
        
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        
        return self.gmm.predict(embeddings)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict probability of belonging to each component.
        
        Args:
            embeddings: Change embeddings
            
        Returns:
            Component probabilities, shape (N, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("GMM must be fitted before prediction")
        
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        
        return self.gmm.predict_proba(embeddings)
    
    def save(self, path: str):
        """Save fitted GMM to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted GMM")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'gmm': self.gmm,
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, path: str):
        """Load fitted GMM from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.gmm = data['gmm']
        self.n_components = data['n_components']
        self.covariance_type = data['covariance_type']
        self.is_fitted = data['is_fitted']
    
    def get_params(self) -> dict:
        """Get GMM parameters."""
        if not self.is_fitted:
            return {}
        
        return {
            'means': self.gmm.means_,
            'covariances': self.gmm.covariances_,
            'weights': self.gmm.weights_
        }


def build_change_gmm(
    n_components: int = 3,
    covariance_type: str = 'full',
    random_state: int = 42
) -> ChangeGMM:
    """
    Factory function to create ChangeGMM.
    
    Args:
        n_components: Number of Gaussian components
        covariance_type: Covariance matrix type
        random_state: Random seed
        
    Returns:
        ChangeGMM instance
    """
    return ChangeGMM(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )


if __name__ == "__main__":
    # Test GMM
    print("Testing ChangeGMM:")
    
    # Generate synthetic embeddings
    np.random.seed(42)
    train_embeddings = np.random.randn(1000, 128)
    test_embeddings = np.random.randn(100, 128)
    
    # Fit GMM
    gmm = build_change_gmm(n_components=3)
    gmm.fit(train_embeddings)
    print(f"GMM fitted: {gmm.is_fitted}")
    
    # Compute log-likelihoods
    train_ll = gmm.log_likelihood(train_embeddings)
    test_ll = gmm.log_likelihood(test_embeddings)
    
    print(f"\nTrain log-likelihood: mean={train_ll.mean():.3f}, std={train_ll.std():.3f}")
    print(f"Test log-likelihood: mean={test_ll.mean():.3f}, std={test_ll.std():.3f}")
    
    # Component prediction
    components = gmm.predict(test_embeddings[:10])
    print(f"\nComponent assignments (first 10): {components}")
    
    # Test with PyTorch tensors
    torch_embeddings = torch.randn(50, 128)
    torch_ll = gmm.log_likelihood(torch_embeddings)
    print(f"\nPyTorch tensor log-likelihood: {torch_ll.mean():.3f}")

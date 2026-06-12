"""
Deep Ensemble for Epistemic Uncertainty Estimation

Implements deep ensembles following Lakshminarayanan et al. (NeurIPS 2017).
Inspired by UnCRtainTS approach for uncertainty in remote sensing.

Multiple models with different initializations provide:
- Predictive mean (ensemble average)
- Predictive variance (ensemble disagreement)
- Epistemic uncertainty estimation
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np
from copy import deepcopy


class DeepEnsemble(nn.Module):
    """
    Deep ensemble wrapper for change detection models.
    
    Trains multiple instances of the same model with different random seeds.
    At inference, aggregates predictions to estimate uncertainty.
    
    Args:
        model_class: Class of the model to ensemble
        model_kwargs: Keyword arguments for model instantiation
        n_models: Number of ensemble members (minimum 3 recommended)
        device: Device for computation
    """
    
    def __init__(
        self,
        model_class: type,
        model_kwargs: dict,
        n_models: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(DeepEnsemble, self).__init__()
        
        if n_models < 2:
            raise ValueError("Ensemble must have at least 2 models")
        
        self.n_models = n_models
        self.device = torch.device(device)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        
        # Create ensemble members
        self.models = nn.ModuleList([
            model_class(**model_kwargs).to(self.device)
            for _ in range(n_models)
        ])
        
        self.is_fitted = [False] * n_models
    
    def forward(
        self,
        *args,
        return_all: bool = False,
        **kwargs
    ):
        """
        Forward pass through all ensemble members.
        
        Args:
            *args: Positional arguments for model forward
            return_all: If True, return predictions from all models
            **kwargs: Keyword arguments for model forward
            
        Returns:
            If return_all=False: (mean_prediction, prediction_variance)
            If return_all=True: (all_predictions, mean_prediction, prediction_variance)
        """
        predictions = []
        
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)
        
        # Stack predictions
        if isinstance(predictions[0], tuple):
            # Handle models that return multiple outputs
            stacked = tuple(torch.stack([p[i] for p in predictions]) for i in range(len(predictions[0])))
            means = tuple(s.mean(dim=0) for s in stacked)
            variances = tuple(s.var(dim=0) for s in stacked)
            
            if return_all:
                return stacked, means, variances
            else:
                return means, variances
        else:
            # Single output
            stacked = torch.stack(predictions)
            mean = stacked.mean(dim=0)
            variance = stacked.var(dim=0)
            
            if return_all:
                return stacked, mean, variance
            else:
                return mean, variance
    
    def predict_with_uncertainty(
        self,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with epistemic uncertainty estimates.
        
        For binary classification, returns:
        - mean_prob: Average predicted probability
        - aleatoric: Aleatoric uncertainty (average predictive entropy)
        - epistemic: Epistemic uncertainty (mutual information)
        
        Args:
            *args: Inputs to the model
            **kwargs: Additional arguments
            
        Returns:
            (mean_prob, aleatoric_uncertainty, epistemic_uncertainty)
        """
        all_logits = []
        
        for model in self.models:
            logits = model(*args, **kwargs)
            all_logits.append(logits)
        
        # Stack and convert to probabilities
        all_logits = torch.stack(all_logits)  # (n_models, batch, 1)
        all_probs = torch.sigmoid(all_logits)
        
        # Mean prediction
        mean_prob = all_probs.mean(dim=0)
        
        # Epistemic uncertainty (variance of predictions)
        epistemic = all_probs.var(dim=0)
        
        # Aleatoric uncertainty (average entropy)
        # For binary classification: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8
        entropies = -(all_probs * torch.log(all_probs + eps) + 
                      (1 - all_probs) * torch.log(1 - all_probs + eps))
        aleatoric = entropies.mean(dim=0)
        
        return mean_prob, aleatoric, epistemic
    
    def get_model(self, index: int) -> nn.Module:
        """Get specific ensemble member."""
        if index >= self.n_models:
            raise IndexError(f"Model index {index} out of range [0, {self.n_models})")
        return self.models[index]
    
    def train_model(self, index: int):
        """Set specific model to training mode."""
        self.models[index].train()
    
    def eval_model(self, index: int):
        """Set specific model to evaluation mode."""
        self.models[index].eval()
    
    def train_all(self):
        """Set all models to training mode."""
        for model in self.models:
            model.train()
    
    def eval_all(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            model.eval()
    
    def save(self, path: str):
        """
        Save all ensemble models.
        
        Args:
            path: Base path (will create path_0.pth, path_1.pth, etc.)
        """
        base_path = path.rsplit('.', 1)[0]
        
        for i, model in enumerate(self.models):
            model_path = f"{base_path}_model_{i}.pth"
            torch.save({
                'model_state': model.state_dict(),
                'model_kwargs': self.model_kwargs,
                'is_fitted': self.is_fitted[i]
            }, model_path)
        
        # Save ensemble metadata
        meta_path = f"{base_path}_meta.pth"
        torch.save({
            'n_models': self.n_models,
            'model_class': self.model_class.__name__,
            'is_fitted': self.is_fitted
        }, meta_path)
    
    def load(self, path: str):
        """
        Load all ensemble models.
        
        Args:
            path: Base path (will load path_0.pth, path_1.pth, etc.)
        """
        base_path = path.rsplit('.', 1)[0]
        
        # Load metadata
        meta_path = f"{base_path}_meta.pth"
        meta = torch.load(meta_path, map_location=self.device)
        
        if meta['n_models'] != self.n_models:
            raise ValueError(f"Ensemble size mismatch: expected {self.n_models}, got {meta['n_models']}")
        
        # Load each model
        for i in range(self.n_models):
            model_path = f"{base_path}_model_{i}.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.models[i].load_state_dict(checkpoint['model_state'])
            self.is_fitted[i] = checkpoint['is_fitted']


class EnsembleTrainer:
    """
    Helper class for training deep ensembles.
    
    Manages training of multiple ensemble members with different seeds.
    
    Args:
        ensemble: DeepEnsemble instance
        base_seed: Base random seed (each model gets base_seed + i)
    """
    
    def __init__(self, ensemble: DeepEnsemble, base_seed: int = 42):
        self.ensemble = ensemble
        self.base_seed = base_seed
        self.seeds = [base_seed + i for i in range(ensemble.n_models)]
    
    def get_seed(self, model_index: int) -> int:
        """Get seed for specific ensemble member."""
        return self.seeds[model_index]
    
    def train_member(
        self,
        model_index: int,
        train_fn,
        *args,
        **kwargs
    ):
        """
        Train a single ensemble member.
        
        Args:
            model_index: Index of model to train
            train_fn: Training function that takes (model, *args, **kwargs)
            *args, **kwargs: Arguments for training function
        """
        # Set seed for this member
        seed = self.get_seed(model_index)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Get model
        model = self.ensemble.get_model(model_index)
        
        # Train
        train_fn(model, *args, **kwargs)
        
        # Mark as fitted
        self.ensemble.is_fitted[model_index] = True


def build_ensemble(
    model_class: type,
    model_kwargs: dict,
    n_models: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> DeepEnsemble:
    """
    Factory function to create DeepEnsemble.
    
    Args:
        model_class: Model class to ensemble
        model_kwargs: Model initialization arguments
        n_models: Number of ensemble members
        device: Computation device
        
    Returns:
        DeepEnsemble instance
    """
    return DeepEnsemble(
        model_class=model_class,
        model_kwargs=model_kwargs,
        n_models=n_models,
        device=device
    )


if __name__ == "__main__":
    print("Testing Deep Ensemble:")
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=1):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    # Create ensemble
    ensemble = build_ensemble(
        model_class=SimpleModel,
        model_kwargs={'input_dim': 10, 'output_dim': 1},
        n_models=3,
        device='cpu'
    )
    
    print(f"Ensemble created with {ensemble.n_models} models")
    
    # Test forward pass
    x = torch.randn(4, 10)
    mean, var = ensemble(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Mean prediction shape: {mean.shape}")
    print(f"Variance shape: {var.shape}")
    print(f"Sample variance: {var[0].item():.4f}")
    
    # Test uncertainty estimation
    mean_prob, aleatoric, epistemic = ensemble.predict_with_uncertainty(x)
    print(f"\nMean probability shape: {mean_prob.shape}")
    print(f"Aleatoric uncertainty: {aleatoric[0].item():.4f}")
    print(f"Epistemic uncertainty: {epistemic[0].item():.4f}")

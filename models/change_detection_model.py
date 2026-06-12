"""
Complete Change Detection System

Integrates all components:
- Siamese ResNet encoder
- Change embedding
- Binary classifier
- Density models (GMM/Flow)
- Three-way decision logic
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Literal
import numpy as np

from .siamese_resnet import SiameseResNet, FeatureDifference
from .change_embedding import ChangeDetectionHead
from .density_models import ChangeGMM, ChangeNormalizingFlow


class ChangeDetectionModel(nn.Module):
    """
    Complete change detection model with OOD awareness.
    
    Architecture:
        Input (I_t, I_t') 
        → Siamese Encoder 
        → Feature Difference
        → Change Embedding (z_Δ)
        → Binary Classifier + Density Model
        → Three-way Decision
    
    Args:
        backbone: 'resnet18' or 'resnet34'
        embedding_dim: Dimension of change embedding (64 or 128)
        density_model_type: 'gmm' or 'normalizing_flow' or None
        n_gmm_components: Number of GMM components
        n_flows: Number of normalizing flow layers
        pretrained: Use ImageNet pretrained weights
        input_channels: Number of input channels
    """
    
    def __init__(
        self,
        backbone: Literal['resnet18', 'resnet34'] = 'resnet18',
        embedding_dim: int = 128,
        density_model_type: Optional[Literal['gmm', 'normalizing_flow']] = 'gmm',
        n_gmm_components: int = 3,
        n_flows: int = 6,
        pretrained: bool = True,
        input_channels: int = 3
    ):
        super(ChangeDetectionModel, self).__init__()
        
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.density_model_type = density_model_type
        
        # Siamese encoder
        self.encoder = SiameseResNet(
            backbone=backbone,
            pretrained=pretrained,
            input_channels=input_channels,
            feature_dim=512
        )
        
        # Feature difference
        self.feature_diff = FeatureDifference()
        
        # Change detection head (embedding + classifier)
        encoder_dim = self.encoder.get_feature_dim()
        self.change_head = ChangeDetectionHead(
            input_dim=encoder_dim,
            embedding_dim=embedding_dim,
            hidden_dims=[256, 256],
            classifier_hidden=64,
            use_gap=True
        )
        
        # Density model (fitted during training)
        self.density_model = None
        if density_model_type == 'gmm':
            self.density_model = ChangeGMM(n_components=n_gmm_components)
        elif density_model_type == 'normalizing_flow':
            self.density_model = ChangeNormalizingFlow(
                input_dim=embedding_dim,
                n_flows=n_flows
            )
        
        # Thresholds for three-way decision (tuned on validation set)
        self.thresholds = {
            'no_change': 0.5,
            'change_confident': 0.7,
            'log_likelihood': -10.0,
            'uncertainty': 0.3
        }
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_embedding: bool = False
    ):
        """
        Forward pass through the model.
        
        Args:
            x1: First image (B, C, H, W)
            x2: Second image (B, C, H, W)
            return_embedding: Return change embedding
            
        Returns:
            If return_embedding=False: logits (B, 1)
            If return_embedding=True: (logits, z_delta)
        """
        # Encode both images
        f1, f2 = self.encoder(x1, x2)
        
        # Compute feature difference
        diff = self.feature_diff(f1, f2)
        
        # Change detection head
        if return_embedding:
            logits, z_delta = self.change_head(diff, return_embedding=True)
            return logits, z_delta
        else:
            logits = self.change_head(diff, return_embedding=False)
            return logits
    
    def predict(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict change probabilities.
        
        Args:
            x1: First image (B, C, H, W)
            x2: Second image (B, C, H, W)
            
        Returns:
            (probabilities, embeddings)
        """
        self.eval()
        with torch.no_grad():
            logits, z_delta = self(x1, x2, return_embedding=True)
            probs = torch.sigmoid(logits)
        
        return probs, z_delta
    
    def compute_log_likelihood(
        self,
        z_delta: torch.Tensor
    ) -> np.ndarray:
        """
        Compute log-likelihood under density model.
        
        Args:
            z_delta: Change embeddings (B, embedding_dim)
            
        Returns:
            Log-likelihoods (B,)
        """
        if self.density_model is None:
            raise RuntimeError("Density model not configured")
        
        if not self.density_model.is_fitted:
            raise RuntimeError("Density model must be fitted first")
        
        return self.density_model.log_likelihood(z_delta)
    
    def fit_density_model(
        self,
        train_embeddings: torch.Tensor,
        verbose: bool = False
    ):
        """
        Fit density model to training change embeddings.
        
        Args:
            train_embeddings: Change embeddings from training change samples
            verbose: Print fitting progress
        """
        if self.density_model is None:
            raise RuntimeError("Density model not configured")
        
        if verbose:
            print(f"Fitting {self.density_model_type} to {len(train_embeddings)} change embeddings...")
        
        self.density_model.fit(train_embeddings, verbose=verbose)
        
        if verbose:
            print("Density model fitted successfully")
    
    def three_way_decision(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make three-way decision: no-change / change (confident) / change (OOD/abstain).
        
        Decision logic:
            if P(no-change) > τ0:
                output = no-change (0)
            elif P(change) > τ1 AND log_likelihood > τ2 AND uncertainty < τ3:
                output = change (confident) (1)
            else:
                output = change (OOD/abstain) (2)
        
        Args:
            x1: First image (B, C, H, W)
            x2: Second image (B, C, H, W)
            uncertainty: Epistemic uncertainty (B,) - optional
            
        Returns:
            (decisions, info_dict)
            decisions: Tensor of shape (B,) with values {0, 1, 2}
            info_dict: Dictionary with probabilities, log-likelihoods, etc.
        """
        self.eval()
        with torch.no_grad():
            probs, z_delta = self.predict(x1, x2)
            probs = probs.squeeze(-1)
            
            # Get log-likelihoods if density model is available
            if self.density_model is not None and self.density_model.is_fitted:
                log_liks = self.compute_log_likelihood(z_delta)
                log_liks = torch.FloatTensor(log_liks).to(probs.device)
            else:
                log_liks = torch.zeros_like(probs)
            
            # Initialize decisions
            batch_size = probs.shape[0]
            decisions = torch.zeros(batch_size, dtype=torch.long, device=probs.device)
            
            # Apply thresholds
            no_change_mask = probs < self.thresholds['no_change']
            decisions[no_change_mask] = 0  # no-change
            
            change_mask = probs >= self.thresholds['change_confident']
            
            if self.density_model is not None and self.density_model.is_fitted:
                in_dist_mask = log_liks > self.thresholds['log_likelihood']
                change_mask = change_mask & in_dist_mask
            
            if uncertainty is not None:
                low_uncertainty_mask = uncertainty < self.thresholds['uncertainty']
                change_mask = change_mask & low_uncertainty_mask
            
            decisions[change_mask] = 1  # change (confident)
            
            # Everything else is abstain
            abstain_mask = ~no_change_mask & ~change_mask
            decisions[abstain_mask] = 2  # change (OOD/abstain)
            
            info = {
                'probabilities': probs,
                'embeddings': z_delta,
                'log_likelihoods': log_liks,
                'uncertainty': uncertainty if uncertainty is not None else torch.zeros_like(probs),
                'no_change_mask': no_change_mask,
                'change_mask': change_mask,
                'abstain_mask': abstain_mask
            }
        
        return decisions, info
    
    def set_thresholds(
        self,
        no_change: Optional[float] = None,
        change_confident: Optional[float] = None,
        log_likelihood: Optional[float] = None,
        uncertainty: Optional[float] = None
    ):
        """
        Set decision thresholds.
        
        Args:
            no_change: Threshold for P(no-change)
            change_confident: Threshold for P(change)
            log_likelihood: Threshold for log-likelihood
            uncertainty: Threshold for epistemic uncertainty
        """
        if no_change is not None:
            self.thresholds['no_change'] = no_change
        if change_confident is not None:
            self.thresholds['change_confident'] = change_confident
        if log_likelihood is not None:
            self.thresholds['log_likelihood'] = log_likelihood
        if uncertainty is not None:
            self.thresholds['uncertainty'] = uncertainty
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return self.thresholds.copy()


def build_change_detection_model(
    backbone: str = 'resnet18',
    embedding_dim: int = 128,
    density_model_type: Optional[str] = 'gmm',
    **kwargs
) -> ChangeDetectionModel:
    """
    Factory function to build complete change detection model.
    
    Args:
        backbone: 'resnet18' or 'resnet34'
        embedding_dim: Change embedding dimension
        density_model_type: 'gmm', 'normalizing_flow', or None
        **kwargs: Additional arguments for model
        
    Returns:
        ChangeDetectionModel instance
    """
    return ChangeDetectionModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        density_model_type=density_model_type,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing Complete Change Detection Model:")
    
    # Create model
    model = build_change_detection_model(
        backbone='resnet18',
        embedding_dim=128,
        density_model_type='gmm',
        pretrained=False
    )
    
    print(f"Model created with backbone: {model.backbone_name}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(f"Density model: {model.density_model_type}")
    
    # Test forward pass
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    logits, z_delta = model(x1, x2, return_embedding=True)
    print(f"\nForward pass:")
    print(f"Input shape: {x1.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Embedding shape: {z_delta.shape}")
    
    # Test prediction
    probs, embeddings = model.predict(x1, x2)
    print(f"\nPrediction:")
    print(f"Probabilities: {probs.squeeze()}")
    
    # Fit density model (with dummy data)
    train_embeddings = torch.randn(100, 128)
    model.fit_density_model(train_embeddings, verbose=True)
    
    # Test log-likelihood
    log_liks = model.compute_log_likelihood(embeddings)
    print(f"\nLog-likelihoods: {log_liks}")
    
    # Test three-way decision
    model.set_thresholds(no_change=0.3, change_confident=0.7, log_likelihood=-15.0)
    decisions, info = model.three_way_decision(x1, x2)
    print(f"\nThree-way decisions: {decisions}")
    print(f"Decision distribution: no-change={(decisions==0).sum()}, "
          f"change={( decisions==1).sum()}, abstain={(decisions==2).sum()}")

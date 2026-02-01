"""
Change Embedding Module

Learns a latent representation z_Δ of how the scene changed.
This embedding captures the nature of change, not just its presence.

Following the problem specification:
    z_Δ = MLP(GlobalAveragePooling(|f_t - f_t'|))
"""

import torch
import torch.nn as nn
from typing import Optional


class ChangeEmbedding(nn.Module):
    """
    Transform feature differences into a latent change embedding.
    
    This module learns to represent HOW a scene changed, encoding the
    type and characteristics of the change in a low-dimensional embedding.
    
    Args:
        input_dim: Dimension of input features (from encoder)
        embedding_dim: Dimension of change embedding (64 or 128)
        hidden_dims: List of hidden layer dimensions for MLP
        use_gap: Whether to apply Global Average Pooling (for spatial features)
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: Optional[list] = None,
        use_gap: bool = True
    ):
        super(ChangeEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.use_gap = use_gap
        
        # Global Average Pooling for spatial features
        if use_gap:
            self.gap = nn.AdaptiveAvgPool2d(1)
        else:
            self.gap = None
        
        # MLP for embedding
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        layers = []
        in_features = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_features = hidden_dim
        
        # Final projection to embedding dimension
        layers.append(nn.Linear(in_features, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, feature_diff: torch.Tensor) -> torch.Tensor:
        """
        Compute change embedding from feature differences.
        
        Args:
            feature_diff: Absolute difference |f_t - f_t'|
                         Shape: (B, C, H, W) if spatial, or (B, C) if already pooled
        
        Returns:
            Change embedding z_Δ of shape (B, embedding_dim)
        """
        # Apply global average pooling if spatial features
        if self.gap is not None and len(feature_diff.shape) == 4:
            x = self.gap(feature_diff)
            x = x.flatten(1)
        else:
            x = feature_diff
        
        # Apply MLP to get embedding
        z_delta = self.mlp(x)
        
        return z_delta


class ChangeClassifier(nn.Module):
    """
    Binary change classifier head operating on change embeddings.
    
    Predicts whether a change occurred (binary classification).
    
    Args:
        embedding_dim: Dimension of input change embedding
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64
    ):
        super(ChangeClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z_delta: torch.Tensor) -> torch.Tensor:
        """
        Predict change probability.
        
        Args:
            z_delta: Change embedding (B, embedding_dim)
            
        Returns:
            Logits (B, 1) - apply sigmoid to get probabilities
        """
        return self.classifier(z_delta)


class ChangeDetectionHead(nn.Module):
    """
    Complete change detection head: embedding + classifier.
    
    Combines the change embedding module and binary classifier
    into a single module for convenience.
    
    Args:
        input_dim: Feature dimension from encoder
        embedding_dim: Change embedding dimension
        hidden_dims: Hidden dimensions for embedding MLP
        classifier_hidden: Hidden dimension for classifier
        use_gap: Apply global average pooling
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: Optional[list] = None,
        classifier_hidden: int = 64,
        use_gap: bool = True
    ):
        super(ChangeDetectionHead, self).__init__()
        
        self.embedding = ChangeEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            use_gap=use_gap
        )
        
        self.classifier = ChangeClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=classifier_hidden
        )
    
    def forward(
        self,
        feature_diff: torch.Tensor,
        return_embedding: bool = False
    ):
        """
        Forward pass through embedding and classifier.
        
        Args:
            feature_diff: Feature difference |f_t - f_t'|
            return_embedding: If True, return both logits and embedding
            
        Returns:
            If return_embedding=False: logits (B, 1)
            If return_embedding=True: (logits, z_delta)
        """
        z_delta = self.embedding(feature_diff)
        logits = self.classifier(z_delta)
        
        if return_embedding:
            return logits, z_delta
        else:
            return logits


if __name__ == "__main__":
    # Test change embedding module
    print("Testing ChangeEmbedding with spatial features:")
    embedding = ChangeEmbedding(input_dim=512, embedding_dim=128, use_gap=True)
    spatial_diff = torch.randn(4, 512, 8, 8)
    z_delta = embedding(spatial_diff)
    print(f"Input shape: {spatial_diff.shape}")
    print(f"Embedding shape: {z_delta.shape}")
    
    print("\nTesting ChangeEmbedding with pooled features:")
    embedding_no_gap = ChangeEmbedding(input_dim=512, embedding_dim=64, use_gap=False)
    pooled_diff = torch.randn(4, 512)
    z_delta2 = embedding_no_gap(pooled_diff)
    print(f"Input shape: {pooled_diff.shape}")
    print(f"Embedding shape: {z_delta2.shape}")
    
    print("\nTesting ChangeClassifier:")
    classifier = ChangeClassifier(embedding_dim=128)
    logits = classifier(z_delta)
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities: {torch.sigmoid(logits)[:2]}")
    
    print("\nTesting Complete ChangeDetectionHead:")
    head = ChangeDetectionHead(input_dim=512, embedding_dim=128)
    logits, embedding = head(spatial_diff, return_embedding=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Embedding shape: {embedding.shape}")

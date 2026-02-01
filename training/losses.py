"""
Loss Functions for Change Detection

Includes:
- Binary Cross Entropy Loss
- Dice Loss
- Combined losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinaryChangeDetectionLoss(nn.Module):
    """
    Binary Cross Entropy Loss for change detection.
    
    Args:
        pos_weight: Weight for positive class (change) to handle class imbalance
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = 'mean'
    ):
        super(BinaryChangeDetectionLoss, self).__init__()
        
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BCE loss.
        
        Args:
            logits: Model predictions (B, 1) or (B, H, W)
            targets: Ground truth labels (B,) or (B, H, W)
            
        Returns:
            Loss value
        """
        # Ensure targets are float
        targets = targets.float()
        
        # Match shapes
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        
        if targets.dim() == 1 and logits.dim() > 1:
            targets = targets.view_as(logits)
        
        return self.criterion(logits, targets)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation-based change detection.
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            logits: Model predictions (B, 1, H, W) or (B, 1)
            targets: Ground truth labels (B, H, W) or (B,)
            
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        if probs.dim() > 2:
            probs = probs.view(probs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
        else:
            if probs.shape[1] == 1:
                probs = probs.squeeze(1)
        
        # Ensure targets are float
        targets = targets.float()
        
        # Compute Dice coefficient
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined BCE and Dice loss.
    
    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        pos_weight: Weight for positive class in BCE
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None
    ):
        super(CombinedLoss, self).__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = BinaryChangeDetectionLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            
        Returns:
            Combined loss
        """
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model predictions (B, 1) or (B, H, W)
            targets: Ground truth labels (B,) or (B, H, W)
            
        Returns:
            Focal loss
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Ensure targets are float
        targets = targets.float()
        
        # Match shapes
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
            probs = probs.squeeze(1)
        
        # Compute BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(
    loss_type: str = 'bce',
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function.
    
    Args:
        loss_type: 'bce', 'dice', 'combined', or 'focal'
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss module
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'bce':
        return BinaryChangeDetectionLoss(**kwargs)
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    print("Testing Loss Functions:")
    
    # Create dummy data
    logits = torch.randn(4, 1)
    targets = torch.randint(0, 2, (4,))
    
    # Test BCE loss
    bce_loss = BinaryChangeDetectionLoss()
    loss_val = bce_loss(logits, targets)
    print(f"BCE Loss: {loss_val.item():.4f}")
    
    # Test Dice loss
    dice_loss = DiceLoss()
    loss_val = dice_loss(logits, targets)
    print(f"Dice Loss: {loss_val.item():.4f}")
    
    # Test Combined loss
    combined_loss = CombinedLoss()
    loss_val = combined_loss(logits, targets)
    print(f"Combined Loss: {loss_val.item():.4f}")
    
    # Test Focal loss
    focal_loss = FocalLoss()
    loss_val = focal_loss(logits, targets)
    print(f"Focal Loss: {loss_val.item():.4f}")
    
    # Test with spatial dimensions
    print("\nTesting with spatial dimensions:")
    logits_spatial = torch.randn(2, 1, 32, 32)
    targets_spatial = torch.randint(0, 2, (2, 32, 32))
    
    loss_val = bce_loss(logits_spatial, targets_spatial)
    print(f"Spatial BCE Loss: {loss_val.item():.4f}")
    
    loss_val = dice_loss(logits_spatial, targets_spatial)
    print(f"Spatial Dice Loss: {loss_val.item():.4f}")

"""
Siamese ResNet Encoder for Change Detection

Following Daudt et al. (IGARSS 2018), this module implements a shared-weight 
Siamese encoder using ResNet-18 or ResNet-34 backbone.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from typing import Tuple, Literal


class SiameseResNet(nn.Module):
    """
    Siamese CNN encoder with shared weights for bi-temporal change detection.
    
    Uses ResNet-18 or ResNet-34 as backbone, removes final classification layers,
    and processes two images through the same encoder.
    
    Args:
        backbone: Either 'resnet18' or 'resnet34'
        pretrained: Whether to use ImageNet pretrained weights
        input_channels: Number of input channels (3 for RGB)
        feature_dim: Output feature dimension after encoder
    """
    
    def __init__(
        self,
        backbone: Literal['resnet18', 'resnet34'] = 'resnet18',
        pretrained: bool = True,
        input_channels: int = 3,
        feature_dim: int = 512
    ):
        super(SiameseResNet, self).__init__()
        
        self.backbone_name = backbone
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone == 'resnet18':
            if pretrained:
                weights = ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            base_model = resnet18(weights=weights)
            self.encoder_dim = 512
        elif backbone == 'resnet34':
            if pretrained:
                weights = ResNet34_Weights.IMAGENET1K_V1
            else:
                weights = None
            base_model = resnet34(weights=weights)
            self.encoder_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'resnet34'")
        
        # Modify first conv layer if input channels != 3
        if input_channels != 3:
            base_model.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Extract encoder (remove avgpool and fc)
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )
        
        # Optional projection to match desired feature_dim
        if feature_dim != self.encoder_dim:
            self.projection = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.encoder_dim, feature_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = None
    
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Feature map (B, feature_dim, H', W') or (B, feature_dim) if projection is used
        """
        features = self.encoder(x)
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode two images through shared-weight encoder.
        
        Args:
            x1: First image (time t) - (B, C, H, W)
            x2: Second image (time t') - (B, C, H, W)
            
        Returns:
            Tuple of (features_t, features_t')
        """
        f1 = self.forward_single(x1)
        f2 = self.forward_single(x2)
        
        return f1, f2
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.feature_dim if self.projection is not None else self.encoder_dim


class FeatureDifference(nn.Module):
    """
    Compute feature difference for change detection.
    
    Following standard practice in change detection literature,
    uses absolute difference: |f_t - f_t'|
    """
    
    def __init__(self):
        super(FeatureDifference, self).__init__()
    
    def forward(
        self, 
        f1: torch.Tensor, 
        f2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute absolute difference between features.
        
        Args:
            f1: Features from first image (B, C, H, W) or (B, C)
            f2: Features from second image (B, C, H, W) or (B, C)
            
        Returns:
            Absolute difference (B, C, H, W) or (B, C)
        """
        return torch.abs(f1 - f2)


def build_siamese_resnet(
    backbone: str = 'resnet18',
    pretrained: bool = True,
    input_channels: int = 3,
    feature_dim: int = 512
) -> SiameseResNet:
    """
    Factory function to build Siamese ResNet encoder.
    
    Args:
        backbone: 'resnet18' or 'resnet34'
        pretrained: Use ImageNet weights
        input_channels: Number of input channels
        feature_dim: Output feature dimension
        
    Returns:
        SiameseResNet model
    """
    return SiameseResNet(
        backbone=backbone,
        pretrained=pretrained,
        input_channels=input_channels,
        feature_dim=feature_dim
    )


if __name__ == "__main__":
    # Test the model
    model = build_siamese_resnet(backbone='resnet18', pretrained=False)
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    f1, f2 = model(x1, x2)
    print(f"Input shape: {x1.shape}")
    print(f"Feature 1 shape: {f1.shape}")
    print(f"Feature 2 shape: {f2.shape}")
    
    diff_module = FeatureDifference()
    diff = diff_module(f1, f2)
    print(f"Difference shape: {diff.shape}")

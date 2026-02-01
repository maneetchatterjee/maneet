"""
Visualization Utilities

Create visualizations for:
- Change maps
- OOD likelihood heatmaps
- Confusion examples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import cv2


def visualize_change_prediction(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Change Detection"
):
    """
    Visualize change detection result.
    
    Args:
        img_t1: First image (H, W, C) or (C, H, W)
        img_t2: Second image (H, W, C) or (C, H, W)
        label: Ground truth label (H, W)
        prediction: Predicted label (H, W)
        save_path: Path to save figure
        title: Figure title
    """
    # Convert to HWC format if needed
    if img_t1.shape[0] == 3:
        img_t1 = np.transpose(img_t1, (1, 2, 0))
    if img_t2.shape[0] == 3:
        img_t2 = np.transpose(img_t2, (1, 2, 0))
    
    # Normalize to [0, 1] if needed
    if img_t1.max() > 1.0:
        img_t1 = img_t1 / 255.0
    if img_t2.max() > 1.0:
        img_t2 = img_t2 / 255.0
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Image t1
    axes[0, 0].imshow(img_t1)
    axes[0, 0].set_title('Image T1')
    axes[0, 0].axis('off')
    
    # Image t2
    axes[0, 1].imshow(img_t2)
    axes[0, 1].set_title('Image T2')
    axes[0, 1].axis('off')
    
    # Ground truth
    axes[1, 0].imshow(label, cmap='gray')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # Prediction
    axes[1, 1].imshow(prediction, cmap='gray')
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_ood_likelihood(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    likelihood_map: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "OOD Likelihood"
):
    """
    Visualize OOD likelihood heatmap.
    
    Args:
        img_t1: First image
        img_t2: Second image
        likelihood_map: Log-likelihood values (H, W)
        prediction: Binary prediction (H, W)
        save_path: Path to save figure
        title: Figure title
    """
    # Convert to HWC format if needed
    if img_t1.shape[0] == 3:
        img_t1 = np.transpose(img_t1, (1, 2, 0))
    if img_t2.shape[0] == 3:
        img_t2 = np.transpose(img_t2, (1, 2, 0))
    
    # Normalize images
    if img_t1.max() > 1.0:
        img_t1 = img_t1 / 255.0
    if img_t2.max() > 1.0:
        img_t2 = img_t2 / 255.0
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Image t1
    axes[0, 0].imshow(img_t1)
    axes[0, 0].set_title('Image T1')
    axes[0, 0].axis('off')
    
    # Image t2
    axes[0, 1].imshow(img_t2)
    axes[0, 1].set_title('Image T2')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[1, 0].imshow(prediction, cmap='gray')
    axes[1, 0].set_title('Change Prediction')
    axes[1, 0].axis('off')
    
    # Likelihood heatmap
    im = axes[1, 1].imshow(likelihood_map, cmap='RdYlGn', vmin=-20, vmax=0)
    axes[1, 1].set_title('Log-Likelihood (Green=In-Dist, Red=OOD)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_three_way_decision(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    label: np.ndarray,
    decision: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Three-Way Decision"
):
    """
    Visualize three-way decision output.
    
    Args:
        img_t1: First image
        img_t2: Second image
        label: Ground truth (H, W)
        decision: Three-way decision (H, W) - 0=no-change, 1=change, 2=abstain
        save_path: Path to save figure
        title: Figure title
    """
    # Convert to HWC format if needed
    if img_t1.shape[0] == 3:
        img_t1 = np.transpose(img_t1, (1, 2, 0))
    if img_t2.shape[0] == 3:
        img_t2 = np.transpose(img_t2, (1, 2, 0))
    
    # Normalize images
    if img_t1.max() > 1.0:
        img_t1 = img_t1 / 255.0
    if img_t2.max() > 1.0:
        img_t2 = img_t2 / 255.0
    
    # Create colored decision map
    # 0 = no-change (blue)
    # 1 = change (green)
    # 2 = abstain (red)
    decision_colored = np.zeros((*decision.shape, 3))
    decision_colored[decision == 0] = [0, 0, 1]  # Blue for no-change
    decision_colored[decision == 1] = [0, 1, 0]  # Green for change
    decision_colored[decision == 2] = [1, 0, 0]  # Red for abstain
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Image t1
    axes[0, 0].imshow(img_t1)
    axes[0, 0].set_title('Image T1')
    axes[0, 0].axis('off')
    
    # Image t2
    axes[0, 1].imshow(img_t2)
    axes[0, 1].set_title('Image T2')
    axes[0, 1].axis('off')
    
    # Ground truth
    axes[1, 0].imshow(label, cmap='gray')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # Three-way decision
    axes[1, 1].imshow(decision_colored)
    axes[1, 1].set_title('Decision (Blue=No-Change, Green=Change, Red=Abstain)')
    axes[1, 1].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[str] = None,
    main_title: str = "Comparison"
):
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of images
        titles: List of titles for each image
        save_path: Path to save figure
        main_title: Main title for figure
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        # Convert to HWC if needed
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        if img.ndim == 2:
            axes[idx].imshow(img, cmap='gray')
        else:
            axes[idx].imshow(img)
        
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("Testing Visualization Utilities:")
    
    # Create dummy data
    img_t1 = np.random.rand(256, 256, 3)
    img_t2 = np.random.rand(256, 256, 3)
    label = np.random.randint(0, 2, (256, 256))
    prediction = np.random.randint(0, 2, (256, 256))
    
    # Test change prediction visualization
    print("Creating change prediction visualization...")
    visualize_change_prediction(
        img_t1, img_t2, label, prediction,
        save_path='/tmp/test_change_pred.png',
        title="Test Change Prediction"
    )
    print("Saved to /tmp/test_change_pred.png")
    
    # Test three-way decision visualization
    decision = np.random.randint(0, 3, (256, 256))
    print("Creating three-way decision visualization...")
    visualize_three_way_decision(
        img_t1, img_t2, label, decision,
        save_path='/tmp/test_three_way.png',
        title="Test Three-Way Decision"
    )
    print("Saved to /tmp/test_three_way.png")

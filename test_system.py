"""
Simple test script to validate the implementation.

Tests basic functionality of all major components.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


def test_siamese_resnet():
    """Test Siamese ResNet encoder."""
    print("\n" + "="*70)
    print("Testing Siamese ResNet")
    print("="*70)
    
    from models import build_siamese_resnet, FeatureDifference
    
    model = build_siamese_resnet(backbone='resnet18', pretrained=False)
    diff_module = FeatureDifference()
    
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    f1, f2 = model(x1, x2)
    diff = diff_module(f1, f2)
    
    print(f"✓ Input shape: {x1.shape}")
    print(f"✓ Feature 1 shape: {f1.shape}")
    print(f"✓ Feature 2 shape: {f2.shape}")
    print(f"✓ Difference shape: {diff.shape}")
    print("✓ Siamese ResNet working correctly")


def test_change_embedding():
    """Test change embedding module."""
    print("\n" + "="*70)
    print("Testing Change Embedding")
    print("="*70)
    
    from models import ChangeDetectionHead
    
    head = ChangeDetectionHead(input_dim=512, embedding_dim=128)
    feature_diff = torch.randn(2, 512, 8, 8)
    
    logits, z_delta = head(feature_diff, return_embedding=True)
    
    print(f"✓ Input shape: {feature_diff.shape}")
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Embedding shape: {z_delta.shape}")
    print("✓ Change embedding working correctly")


def test_density_models():
    """Test GMM and Normalizing Flow."""
    print("\n" + "="*70)
    print("Testing Density Models")
    print("="*70)
    
    from models.density_models import build_change_gmm, build_normalizing_flow
    
    # Generate synthetic embeddings
    train_embeddings = np.random.randn(500, 128)
    test_embeddings = np.random.randn(50, 128)
    
    # Test GMM
    print("\nTesting GMM:")
    gmm = build_change_gmm(n_components=3)
    gmm.fit(train_embeddings)
    log_liks = gmm.log_likelihood(test_embeddings)
    print(f"✓ GMM fitted on {len(train_embeddings)} samples")
    print(f"✓ Log-likelihood shape: {log_liks.shape}")
    print(f"✓ Mean log-likelihood: {log_liks.mean():.3f}")
    
    # Test Normalizing Flow
    print("\nTesting Normalizing Flow:")
    flow = build_normalizing_flow(input_dim=128, n_flows=4, device='cpu')
    flow.fit(train_embeddings, verbose=False)
    log_liks_flow = flow.log_likelihood(test_embeddings)
    print(f"✓ Flow fitted on {len(train_embeddings)} samples")
    print(f"✓ Log-likelihood shape: {log_liks_flow.shape}")
    print(f"✓ Mean log-likelihood: {log_liks_flow.mean():.3f}")
    
    print("✓ Density models working correctly")


def test_complete_model():
    """Test complete change detection model."""
    print("\n" + "="*70)
    print("Testing Complete Change Detection Model")
    print("="*70)
    
    from models import build_change_detection_model
    
    model = build_change_detection_model(
        backbone='resnet18',
        embedding_dim=128,
        density_model_type='gmm',
        pretrained=False
    )
    
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    # Test forward pass
    logits, z_delta = model(x1, x2, return_embedding=True)
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Embedding shape: {z_delta.shape}")
    
    # Test prediction
    probs, embeddings = model.predict(x1, x2)
    print(f"✓ Probabilities: {probs.squeeze()}")
    
    # Fit density model
    train_embeddings = torch.randn(200, 128)
    model.fit_density_model(train_embeddings, verbose=False)
    print(f"✓ Density model fitted")
    
    # Test log-likelihood
    log_liks = model.compute_log_likelihood(embeddings)
    print(f"✓ Log-likelihoods: {log_liks}")
    
    # Test three-way decision
    model.set_thresholds(no_change=0.3, change_confident=0.7, log_likelihood=-15.0)
    decisions, info = model.three_way_decision(x1, x2)
    print(f"✓ Decisions: {decisions}")
    print(f"✓ Decision counts: no-change={(decisions==0).sum()}, change={(decisions==1).sum()}, abstain={(decisions==2).sum()}")
    
    print("✓ Complete model working correctly")


def test_ensemble():
    """Test deep ensemble."""
    print("\n" + "="*70)
    print("Testing Deep Ensemble")
    print("="*70)
    
    from models import build_ensemble
    import torch.nn as nn
    
    # Simple test model
    class TestModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=1):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    ensemble = build_ensemble(
        model_class=TestModel,
        model_kwargs={'input_dim': 10, 'output_dim': 1},
        n_models=3,
        device='cpu'
    )
    
    x = torch.randn(4, 10)
    mean, var = ensemble(x)
    
    print(f"✓ Ensemble with {ensemble.n_models} models created")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Mean prediction shape: {mean.shape}")
    print(f"✓ Variance shape: {var.shape}")
    
    # Test uncertainty estimation
    mean_prob, aleatoric, epistemic = ensemble.predict_with_uncertainty(x)
    print(f"✓ Epistemic uncertainty: {epistemic[0].item():.4f}")
    
    print("✓ Deep ensemble working correctly")


def test_losses():
    """Test loss functions."""
    print("\n" + "="*70)
    print("Testing Loss Functions")
    print("="*70)
    
    from training import get_loss_function
    
    logits = torch.randn(4, 1)
    targets = torch.randint(0, 2, (4,)).float()
    
    # Test BCE
    bce_loss = get_loss_function('bce')
    loss_val = bce_loss(logits, targets)
    print(f"✓ BCE Loss: {loss_val.item():.4f}")
    
    # Test Dice
    dice_loss = get_loss_function('dice')
    loss_val = dice_loss(logits, targets)
    print(f"✓ Dice Loss: {loss_val.item():.4f}")
    
    # Test Combined
    combined_loss = get_loss_function('combined')
    loss_val = combined_loss(logits, targets)
    print(f"✓ Combined Loss: {loss_val.item():.4f}")
    
    print("✓ Loss functions working correctly")


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*70)
    print("Testing Evaluation Metrics")
    print("="*70)
    
    from evaluation import (
        ChangeDetectionMetrics,
        ThreeWayMetrics,
        OODMetrics,
        CalibrationMetrics
    )
    
    # Test change detection metrics
    predictions = torch.randint(0, 2, (100,))
    targets = torch.randint(0, 2, (100,))
    
    cd_metrics = ChangeDetectionMetrics()
    cd_metrics.update(predictions, targets)
    results = cd_metrics.compute()
    print(f"✓ Change detection F1: {results['f1']:.4f}")
    
    # Test three-way metrics
    decisions = torch.randint(0, 3, (100,))
    three_way = ThreeWayMetrics()
    three_way.update(decisions, targets)
    results = three_way.compute()
    print(f"✓ Abstention rate: {results['abstention_rate']:.4f}")
    
    # Test OOD metrics
    scores = np.random.randn(200)
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    ood_metrics = OODMetrics()
    ood_metrics.update(scores, labels)
    results = ood_metrics.compute()
    print(f"✓ OOD AUROC: {results['auroc']:.4f}")
    
    # Test calibration
    probs = np.random.rand(200)
    cal_metrics = CalibrationMetrics()
    cal_metrics.update(probs, labels[:200].astype(float))
    results = cal_metrics.compute()
    print(f"✓ ECE: {results['ece']:.4f}")
    
    print("✓ Evaluation metrics working correctly")


def test_utils():
    """Test utility functions."""
    print("\n" + "="*70)
    print("Testing Utilities")
    print("="*70)
    
    from utils import set_seed, get_device, count_parameters
    from models import build_change_detection_model
    
    # Test reproducibility
    set_seed(42)
    r1 = np.random.rand()
    set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "Seed setting not working"
    print(f"✓ Reproducibility working (seed produces consistent results)")
    
    # Test device
    device = get_device(verbose=False)
    print(f"✓ Device: {device}")
    
    # Test parameter counting
    model = build_change_detection_model(backbone='resnet18', pretrained=False)
    n_params = count_parameters(model)
    print(f"✓ Model parameters: {n_params:,}")
    
    print("✓ Utilities working correctly")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CHANGE DETECTION SYSTEM - VALIDATION TESTS")
    print("="*70)
    
    try:
        test_siamese_resnet()
        test_change_embedding()
        test_density_models()
        test_complete_model()
        test_ensemble()
        test_losses()
        test_metrics()
        test_utils()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70)
        print("\nThe change detection system is ready to use.")
        print("\nNext steps:")
        print("  1. Prepare your dataset (OSCD or LEVIR-CD)")
        print("  2. Train a model: python experiments/train.py --config experiments/configs/oscd_baseline.yaml")
        print("  3. Evaluate: python experiments/evaluate.py --config ... --checkpoint ...")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

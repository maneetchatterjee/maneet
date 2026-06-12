"""
Evaluation Script for Change Detection

Evaluates trained models on test set with:
- Change detection metrics
- OOD detection metrics
- Calibration metrics
- Three-way decision evaluation
"""

import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_change_detection_model
from datasets import OSCDDataset, LEVIRCDDataset, get_levir_transforms
from evaluation import (
    ChangeDetectionMetrics,
    ThreeWayMetrics,
    OODMetrics,
    CalibrationMetrics,
    RiskCoverageMetrics
)
from utils import setup_reproducibility, get_device, setup_logger


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config: dict, split: str):
    """Create dataset based on configuration."""
    dataset_config = config['dataset']
    dataset_name = dataset_config['name'].lower()
    
    if dataset_name == 'oscd':
        dataset = OSCDDataset(
            root_dir=dataset_config['root_dir'],
            split=split,
            transform=None,
            normalize=dataset_config.get('normalize', True),
            use_rgb_only=dataset_config.get('use_rgb_only', True),
            patch_size=None  # Don't use patches for evaluation
        )
    elif dataset_name == 'levir_cd':
        transform = get_levir_transforms(split)
        dataset = LEVIRCDDataset(
            root_dir=dataset_config['root_dir'],
            split=split,
            transform=transform,
            normalize=dataset_config.get('normalize', True),
            patch_size=None
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def evaluate_change_detection(
    model,
    test_loader,
    device,
    logger
):
    """
    Evaluate binary change detection.
    
    Returns:
        Dictionary with metrics
    """
    logger.info("Evaluating binary change detection...")
    
    model.eval()
    metrics = ChangeDetectionMetrics()
    
    with torch.no_grad():
        for batch in test_loader:
            img_t1 = batch['image_t1'].to(device)
            img_t2 = batch['image_t2'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions
            probs, _ = model.predict(img_t1, img_t2)
            predictions = (probs > 0.5).long()
            
            # Update metrics
            metrics.update(predictions, labels)
    
    results = metrics.compute()
    
    logger.info("Change Detection Metrics:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return results


def evaluate_ood_detection(
    model,
    test_loader,
    device,
    logger
):
    """
    Evaluate OOD detection using likelihood scores.
    
    Returns:
        Dictionary with OOD metrics
    """
    if model.density_model is None or not model.density_model.is_fitted:
        logger.warning("Density model not available, skipping OOD evaluation")
        return {}
    
    logger.info("Evaluating OOD detection...")
    
    model.eval()
    ood_metrics = OODMetrics()
    
    with torch.no_grad():
        for batch in test_loader:
            img_t1 = batch['image_t1'].to(device)
            img_t2 = batch['image_t2'].to(device)
            labels = batch['label'].to(device)
            
            # Get embeddings and likelihoods
            probs, z_delta = model.predict(img_t1, img_t2)
            
            # Only evaluate on predicted changes
            pred_change_mask = probs.squeeze() > 0.5
            
            if pred_change_mask.sum() == 0:
                continue
            
            # Get log-likelihoods for predicted changes
            change_embeddings = z_delta[pred_change_mask]
            log_liks = model.compute_log_likelihood(change_embeddings)
            
            # Assume actual changes are in-distribution (label=1)
            true_labels = labels[pred_change_mask]
            
            # Update metrics (higher likelihood = in-distribution)
            ood_metrics.update(log_liks, true_labels.cpu().numpy())
    
    results = ood_metrics.compute()
    
    logger.info("OOD Detection Metrics:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return results


def evaluate_calibration(
    model,
    test_loader,
    device,
    logger
):
    """
    Evaluate model calibration.
    
    Returns:
        Dictionary with calibration metrics
    """
    logger.info("Evaluating calibration...")
    
    model.eval()
    cal_metrics = CalibrationMetrics(n_bins=10)
    
    with torch.no_grad():
        for batch in test_loader:
            img_t1 = batch['image_t1'].to(device)
            img_t2 = batch['image_t2'].to(device)
            labels = batch['label'].to(device)
            
            # Get probabilities
            probs, _ = model.predict(img_t1, img_t2)
            
            # Update metrics
            cal_metrics.update(probs.squeeze().cpu().numpy(), labels.cpu().numpy())
    
    results = cal_metrics.compute()
    
    logger.info("Calibration Metrics:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return results


def evaluate_three_way_decision(
    model,
    test_loader,
    device,
    config,
    logger
):
    """
    Evaluate three-way decision (no-change / change / abstain).
    
    Returns:
        Dictionary with three-way metrics
    """
    logger.info("Evaluating three-way decision...")
    
    # Set thresholds
    thresholds = config.get('thresholds', {})
    model.set_thresholds(
        no_change=thresholds.get('no_change', 0.5),
        change_confident=thresholds.get('change_confident', 0.7),
        log_likelihood=thresholds.get('log_likelihood', -10.0),
        uncertainty=thresholds.get('uncertainty', 0.3)
    )
    
    model.eval()
    three_way_metrics = ThreeWayMetrics()
    
    with torch.no_grad():
        for batch in test_loader:
            img_t1 = batch['image_t1'].to(device)
            img_t2 = batch['image_t2'].to(device)
            labels = batch['label'].to(device)
            
            # Get three-way decisions
            decisions, _ = model.three_way_decision(img_t1, img_t2)
            
            # Update metrics
            three_way_metrics.update(decisions, labels)
    
    results = three_way_metrics.compute()
    
    logger.info("Three-Way Decision Metrics:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return results


def evaluate_risk_coverage(
    model,
    test_loader,
    device,
    logger
):
    """
    Evaluate risk-coverage trade-off.
    
    Returns:
        Dictionary with coverage and risk arrays
    """
    logger.info("Evaluating risk-coverage trade-off...")
    
    model.eval()
    rc_metrics = RiskCoverageMetrics()
    
    with torch.no_grad():
        for batch in test_loader:
            img_t1 = batch['image_t1'].to(device)
            img_t2 = batch['image_t2'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions and confidence
            probs, _ = model.predict(img_t1, img_t2)
            predictions = (probs > 0.5).long()
            
            # Use probability as confidence
            confidence = torch.maximum(probs, 1 - probs).squeeze()
            
            # Update metrics
            rc_metrics.update(
                confidence.cpu().numpy(),
                predictions.cpu().numpy(),
                labels.cpu().numpy()
            )
    
    coverage, risk = rc_metrics.compute_risk_coverage_curve()
    
    logger.info("Risk-Coverage (first 5 points):")
    for c, r in zip(coverage[:5], risk[:5]):
        logger.info(f"  Coverage: {c:.3f}, Risk: {r:.4f}")
    
    return {
        'coverage': coverage.tolist(),
        'risk': risk.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate change detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup reproducibility
    seed = config['experiment'].get('seed', 42)
    setup_reproducibility(seed=seed)
    
    # Setup logger
    logger = setup_logger('evaluation')
    
    # Device
    device = get_device(verbose=True)
    
    # Create test dataset
    logger.info("Creating test dataset...")
    test_dataset = create_dataset(config, 'test')
    logger.info(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model_config = config['model']
    model = build_change_detection_model(**model_config)
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Model loaded successfully")
    
    # Run evaluations
    all_results = {}
    
    # 1. Binary change detection
    cd_results = evaluate_change_detection(model, test_loader, device, logger)
    all_results['change_detection'] = cd_results
    
    # 2. OOD detection
    eval_config = config.get('evaluation', {})
    if eval_config.get('compute_ood', True):
        ood_results = evaluate_ood_detection(model, test_loader, device, logger)
        all_results['ood_detection'] = ood_results
    
    # 3. Calibration
    if eval_config.get('compute_calibration', True):
        cal_results = evaluate_calibration(model, test_loader, device, logger)
        all_results['calibration'] = cal_results
    
    # 4. Three-way decision
    three_way_results = evaluate_three_way_decision(model, test_loader, device, config, logger)
    all_results['three_way_decision'] = three_way_results
    
    # 5. Risk-coverage
    rc_results = evaluate_risk_coverage(model, test_loader, device, logger)
    all_results['risk_coverage'] = rc_results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()

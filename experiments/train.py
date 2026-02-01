"""
Training Script for Change Detection

Main script for training change detection models with OOD awareness.
"""

import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_change_detection_model, build_ensemble
from datasets import OSCDDataset, LEVIRCDDataset, get_levir_transforms
from training import get_loss_function, ChangeDetectionTrainer
from utils import setup_reproducibility, get_device, ExperimentLogger, print_model_summary
import numpy as np


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config: dict, split: str):
    """
    Create dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
    
    Returns:
        Dataset instance
    """
    dataset_config = config['dataset']
    dataset_name = dataset_config['name'].lower()
    
    if dataset_name == 'oscd':
        dataset = OSCDDataset(
            root_dir=dataset_config['root_dir'],
            split=split,
            transform=None,  # TODO: Add transforms
            normalize=dataset_config.get('normalize', True),
            use_rgb_only=dataset_config.get('use_rgb_only', True),
            patch_size=dataset_config.get('patch_size', 256) if split == 'train' else None
        )
    elif dataset_name == 'levir_cd':
        transform = get_levir_transforms(split)
        dataset = LEVIRCDDataset(
            root_dir=dataset_config['root_dir'],
            split=split,
            transform=transform,
            normalize=dataset_config.get('normalize', True),
            patch_size=dataset_config.get('patch_size', 256) if split == 'train' else None
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_model(config: dict, device: torch.device):
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
    
    Returns:
        Model instance
    """
    model_config = config['model']
    ensemble_config = config.get('ensemble', {})
    
    if ensemble_config.get('enabled', False):
        # Create ensemble
        from models import ChangeDetectionModel
        
        model = build_ensemble(
            model_class=ChangeDetectionModel,
            model_kwargs=model_config,
            n_models=ensemble_config['n_models'],
            device=str(device)
        )
    else:
        # Single model
        model = build_change_detection_model(**model_config)
        model = model.to(device)
    
    return model


def create_optimizer(model, config: dict):
    """Create optimizer based on configuration."""
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type'].lower()
    
    params = model.parameters()
    
    if opt_type == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0)
        )
    elif opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=opt_config['lr'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    return optimizer


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler based on configuration."""
    sched_config = config['training']['scheduler']
    sched_type = sched_config['type'].lower()
    
    epochs = config['training']['epochs']
    
    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=sched_config.get('min_lr', 1e-6)
        )
    elif sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 30),
            gamma=sched_config.get('gamma', 0.1)
        )
    elif sched_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config.get('factor', 0.1),
            patience=sched_config.get('patience', 10)
        )
    else:
        scheduler = None
    
    return scheduler


def fit_density_model(model, train_loader, config: dict, logger):
    """
    Fit density model on training change embeddings.
    
    Args:
        model: Change detection model
        train_loader: Training data loader
        config: Configuration dictionary
        logger: Logger instance
    """
    density_config = config.get('density_model_training', {})
    
    if not density_config.get('enabled', True):
        logger.info("Density model fitting disabled")
        return
    
    if model.density_model is None:
        logger.info("No density model configured")
        return
    
    logger.info("Collecting change embeddings from training set...")
    
    model.eval()
    change_embeddings = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in train_loader:
            img_t1 = batch['image_t1'].to(device)
            img_t2 = batch['image_t2'].to(device)
            labels = batch['label'].to(device)
            
            # Get embeddings
            _, z_delta = model(img_t1, img_t2, return_embedding=True)
            
            # Filter only change samples
            change_mask = labels.flatten() == 1
            if change_mask.sum() > 0:
                change_emb = z_delta[change_mask]
                change_embeddings.append(change_emb.cpu())
    
    if len(change_embeddings) == 0:
        logger.warning("No change samples found in training set!")
        return
    
    change_embeddings = torch.cat(change_embeddings, dim=0)
    logger.info(f"Collected {len(change_embeddings)} change embeddings")
    
    min_samples = density_config.get('min_samples', 100)
    if len(change_embeddings) < min_samples:
        logger.warning(f"Too few change samples ({len(change_embeddings)} < {min_samples})")
        return
    
    # Fit density model
    logger.info("Fitting density model...")
    model.fit_density_model(change_embeddings, verbose=True)
    logger.info("Density model fitted successfully")


def main():
    parser = argparse.ArgumentParser(description='Train change detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup reproducibility
    seed = config['experiment'].get('seed', 42)
    setup_reproducibility(seed=seed, benchmark=False)
    
    # Setup experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=config['experiment']['name'],
        output_dir=config['experiment']['output_dir']
    )
    
    exp_logger.log_hyperparameters(config)
    
    # Device
    device = get_device(verbose=True)
    
    # Create datasets
    exp_logger.logger.info("Creating datasets...")
    train_dataset = create_dataset(config, 'train')
    val_dataset = create_dataset(config, 'val')
    
    exp_logger.logger.info(f"Train samples: {len(train_dataset)}")
    exp_logger.logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    exp_logger.logger.info("Creating model...")
    model = create_model(config, device)
    print_model_summary(model)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    loss_config = config['training']['loss']
    criterion = get_loss_function(**loss_config)
    
    # Create trainer
    checkpoint_dir = exp_logger.exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    trainer = ChangeDetectionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        use_amp=config['training'].get('mixed_precision', True),
        grad_clip=config['training'].get('gradient_clip', 1.0),
        scheduler=scheduler,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=10
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        exp_logger.logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    exp_logger.logger.info("Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        save_best=config['training']['checkpoint'].get('save_best', True)
    )
    
    # Fit density model after training
    exp_logger.logger.info("Fitting density model on final model...")
    fit_density_model(model, train_loader, config, exp_logger.logger)
    
    # Save final model with density model
    final_checkpoint = checkpoint_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_checkpoint)
    exp_logger.logger.info(f"Final model saved to {final_checkpoint}")
    
    # Finish experiment
    exp_logger.finish()
    
    print("\nTraining completed successfully!")
    print(f"Output directory: {exp_logger.exp_dir}")


if __name__ == '__main__':
    main()

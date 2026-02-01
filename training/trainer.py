"""
Trainer for Change Detection Model

Implements training loop with:
- Mixed precision (AMP)
- Gradient clipping
- Learning rate scheduling
- Checkpoint management
- TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Callable
from tqdm import tqdm
import time


class ChangeDetectionTrainer:
    """
    Trainer for change detection models.
    
    Args:
        model: Change detection model
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device for training
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping value (None to disable)
        scheduler: Learning rate scheduler (optional)
        checkpoint_dir: Directory to save checkpoints
        log_interval: Log every N batches
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
        scheduler: Optional[object] = None,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 10
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.log_interval = log_interval
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        # Mixed precision scaler
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Checkpoint management
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.train_history = []
        self.val_history = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            img_t1 = batch['image_t1'].to(self.device)
            img_t2 = batch['image_t2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    logits = self.model(img_t1, img_t2)
                    loss = self.criterion(logits, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(img_t1, img_t2)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Epoch metrics
        epoch_loss = running_loss / num_batches
        
        return {'loss': epoch_loss}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                img_t1 = batch['image_t1'].to(self.device)
                img_t2 = batch['image_t2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        logits = self.model(img_t1, img_t2)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(img_t1, img_t2)
                    loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
        
        # Validation metrics
        val_loss = running_loss / num_batches
        
        return {'loss': val_loss}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_best: bool = True
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            save_best: Save best model based on validation loss
        """
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.val_history.append(val_metrics)
                
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f} - "
                      f"Val Loss: {val_metrics['loss']:.4f} - "
                      f"Time: {time.time() - start_time:.2f}s")
                
                # Save best model
                if save_best and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint('best_model.pth')
                        print(f"Best model saved (val_loss: {self.best_val_loss:.4f})")
            else:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f} - "
                      f"Time: {time.time() - start_time:.2f}s")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader is not None:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save periodic checkpoint
            if self.checkpoint_dir is not None and epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
    
    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            raise RuntimeError("Checkpoint directory not set")
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")


if __name__ == "__main__":
    # Test trainer
    print("Testing ChangeDetectionTrainer:")
    
    from ..models import build_change_detection_model
    from .losses import BinaryChangeDetectionLoss
    
    # Create dummy model
    model = build_change_detection_model(
        backbone='resnet18',
        embedding_dim=64,
        pretrained=False
    )
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = BinaryChangeDetectionLoss()
    
    # Create trainer
    trainer = ChangeDetectionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device='cpu',
        use_amp=False
    )
    
    print("Trainer created successfully")
    print(f"Device: {trainer.device}")
    print(f"Mixed precision: {trainer.use_amp}")

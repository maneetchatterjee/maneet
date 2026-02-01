"""
Change Detection Metrics

Implements standard metrics for binary change detection:
- Precision, Recall, F1
- Confusion Matrix
- IoU (Intersection over Union)
- Overall Accuracy
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


class ChangeDetectionMetrics:
    """
    Compute change detection metrics.
    
    Tracks predictions and ground truth, then computes metrics.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated predictions and targets."""
        self.all_predictions = []
        self.all_targets = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update with new predictions and targets.
        
        Args:
            predictions: Binary predictions (0 or 1) or probabilities
            targets: Ground truth labels (0 or 1)
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        # Flatten
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Convert probabilities to binary if needed
        if predictions.dtype == np.float32 or predictions.dtype == np.float64:
            if predictions.max() <= 1.0:
                predictions = (predictions > 0.5).astype(np.int64)
        
        self.all_predictions.append(predictions)
        self.all_targets.append(targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with metrics
        """
        if len(self.all_predictions) == 0:
            raise RuntimeError("No predictions to compute metrics from")
        
        # Concatenate all predictions and targets
        preds = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        
        # Compute metrics
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        accuracy = accuracy_score(targets, preds)
        
        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'iou': float(iou),
            'specificity': float(specificity),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        if len(self.all_predictions) == 0:
            raise RuntimeError("No predictions to compute confusion matrix from")
        
        preds = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        
        return confusion_matrix(targets, preds)


def compute_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics directly from logits.
    
    Args:
        logits: Model output logits
        targets: Ground truth labels
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Dictionary with metrics
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    
    # Convert to binary predictions
    predictions = (probs > threshold).long()
    
    # Compute metrics
    metrics = ChangeDetectionMetrics()
    metrics.update(predictions, targets)
    
    return metrics.compute()


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        predictions: Binary predictions
        targets: Ground truth labels
        
    Returns:
        Dictionary with per-class metrics
    """
    # Class 0: No change
    no_change_mask = targets == 0
    no_change_correct = np.sum((predictions == 0) & no_change_mask)
    no_change_total = np.sum(no_change_mask)
    no_change_acc = no_change_correct / no_change_total if no_change_total > 0 else 0.0
    
    # Class 1: Change
    change_mask = targets == 1
    change_correct = np.sum((predictions == 1) & change_mask)
    change_total = np.sum(change_mask)
    change_acc = change_correct / change_total if change_total > 0 else 0.0
    
    return {
        'no_change': {
            'accuracy': float(no_change_acc),
            'count': int(no_change_total),
            'correct': int(no_change_correct)
        },
        'change': {
            'accuracy': float(change_acc),
            'count': int(change_total),
            'correct': int(change_correct)
        }
    }


class ThreeWayMetrics:
    """
    Metrics for three-way decision (no-change / change / abstain).
    
    Tracks:
    - Accuracy on confident predictions
    - Abstention rate
    - Coverage (1 - abstention rate)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.all_decisions = []
        self.all_targets = []
    
    def update(
        self,
        decisions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update with three-way decisions.
        
        Args:
            decisions: Three-way decisions (0=no-change, 1=change, 2=abstain)
            targets: Ground truth (0=no-change, 1=change)
        """
        if torch.is_tensor(decisions):
            decisions = decisions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        self.all_decisions.append(decisions.flatten())
        self.all_targets.append(targets.flatten())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute three-way metrics.
        
        Returns:
            Dictionary with metrics
        """
        if len(self.all_decisions) == 0:
            raise RuntimeError("No decisions to compute metrics from")
        
        decisions = np.concatenate(self.all_decisions)
        targets = np.concatenate(self.all_targets)
        
        total = len(decisions)
        
        # Abstention rate
        abstain_mask = decisions == 2
        abstention_rate = np.sum(abstain_mask) / total
        coverage = 1.0 - abstention_rate
        
        # Accuracy on confident predictions (non-abstained)
        confident_mask = ~abstain_mask
        if np.sum(confident_mask) > 0:
            confident_decisions = decisions[confident_mask]
            confident_targets = targets[confident_mask]
            confident_accuracy = np.mean(confident_decisions == confident_targets)
        else:
            confident_accuracy = 0.0
        
        # Overall accuracy (treating abstain as wrong)
        overall_accuracy = np.mean(decisions == targets)
        
        # Breakdown by decision type
        no_change_decisions = np.sum(decisions == 0)
        change_decisions = np.sum(decisions == 1)
        abstain_decisions = np.sum(decisions == 2)
        
        return {
            'abstention_rate': float(abstention_rate),
            'coverage': float(coverage),
            'confident_accuracy': float(confident_accuracy),
            'overall_accuracy': float(overall_accuracy),
            'no_change_decisions': int(no_change_decisions),
            'change_decisions': int(change_decisions),
            'abstain_decisions': int(abstain_decisions)
        }


if __name__ == "__main__":
    print("Testing Change Detection Metrics:")
    
    # Create dummy predictions and targets
    predictions = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    
    # Compute metrics
    metrics = ChangeDetectionMetrics()
    metrics.update(predictions, targets)
    results = metrics.compute()
    
    print("\nBinary Change Detection Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print("\nConfusion Matrix:")
    print(metrics.get_confusion_matrix())
    
    # Test three-way metrics
    print("\nTesting Three-Way Metrics:")
    decisions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 1, 2])
    targets = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    
    three_way = ThreeWayMetrics()
    three_way.update(decisions, targets)
    results = three_way.compute()
    
    for key, value in results.items():
        print(f"  {key}: {value}")

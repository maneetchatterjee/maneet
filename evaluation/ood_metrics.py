"""
Out-of-Distribution Detection Metrics

Implements metrics for evaluating OOD detection:
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
- FPR at TPR thresholds
- Likelihood histograms
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


class OODMetrics:
    """
    Metrics for out-of-distribution detection.
    
    Evaluates how well likelihood scores separate in-distribution
    from out-of-distribution samples.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated scores and labels."""
        self.all_scores = []
        self.all_labels = []
    
    def update(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ):
        """
        Update with scores and labels.
        
        Args:
            scores: Likelihood scores or confidence scores
            labels: Binary labels (1 = in-distribution, 0 = OOD)
        """
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        
        self.all_scores.append(scores.flatten())
        self.all_labels.append(labels.flatten())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute OOD detection metrics.
        
        Returns:
            Dictionary with AUROC, AUPR, and FPR metrics
        """
        if len(self.all_scores) == 0:
            raise RuntimeError("No scores to compute metrics from")
        
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        
        # AUROC (higher scores should indicate in-distribution)
        auroc = roc_auc_score(labels, scores)
        
        # AUPR (Average Precision)
        aupr = average_precision_score(labels, scores)
        
        # ROC curve for FPR at specific TPR
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # FPR at 95% TPR
        idx_95 = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95tpr = fpr[idx_95]
        
        # FPR at 90% TPR
        idx_90 = np.argmin(np.abs(tpr - 0.90))
        fpr_at_90tpr = fpr[idx_90]
        
        return {
            'auroc': float(auroc),
            'aupr': float(aupr),
            'fpr_at_95tpr': float(fpr_at_95tpr),
            'fpr_at_90tpr': float(fpr_at_90tpr)
        }
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.
        
        Returns:
            (fpr, tpr, thresholds)
        """
        if len(self.all_scores) == 0:
            raise RuntimeError("No scores to compute ROC curve from")
        
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        
        return roc_curve(labels, scores)
    
    def get_pr_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data.
        
        Returns:
            (precision, recall, thresholds)
        """
        if len(self.all_scores) == 0:
            raise RuntimeError("No scores to compute PR curve from")
        
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        
        return precision_recall_curve(labels, scores)
    
    def plot_likelihood_histogram(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot histogram of likelihood scores for in-dist vs OOD.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        if len(self.all_scores) == 0:
            raise RuntimeError("No scores to plot")
        
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        
        in_dist_scores = scores[labels == 1]
        ood_scores = scores[labels == 0]
        
        plt.figure(figsize=(10, 6))
        plt.hist(in_dist_scores, bins=50, alpha=0.5, label='In-Distribution', density=True)
        plt.hist(ood_scores, bins=50, alpha=0.5, label='Out-of-Distribution', density=True)
        plt.xlabel('Log-Likelihood Score')
        plt.ylabel('Density')
        plt.title('Likelihood Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def compute_likelihood_threshold(
    in_dist_scores: np.ndarray,
    fpr_target: float = 0.05
) -> float:
    """
    Compute likelihood threshold for a target false positive rate.
    
    Args:
        in_dist_scores: Likelihood scores from in-distribution samples
        fpr_target: Target false positive rate (e.g., 0.05 for 5% FPR)
        
    Returns:
        Threshold value
    """
    # Sort scores in descending order
    sorted_scores = np.sort(in_dist_scores)
    
    # Find threshold at desired FPR
    idx = int(fpr_target * len(sorted_scores))
    threshold = sorted_scores[idx]
    
    return threshold


class CoverageAccuracyMetrics:
    """
    Metrics for coverage vs accuracy trade-off.
    
    Evaluates how accuracy changes as coverage (1 - abstention rate) varies.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset data."""
        self.all_confidences = []
        self.all_predictions = []
        self.all_targets = []
    
    def update(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """
        Update with confidences, predictions, and targets.
        
        Args:
            confidences: Confidence scores (higher = more confident)
            predictions: Model predictions
            targets: Ground truth labels
        """
        if torch.is_tensor(confidences):
            confidences = confidences.detach().cpu().numpy()
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        self.all_confidences.append(confidences.flatten())
        self.all_predictions.append(predictions.flatten())
        self.all_targets.append(targets.flatten())
    
    def compute_coverage_accuracy_curve(
        self,
        n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coverage vs accuracy curve.
        
        Args:
            n_points: Number of points on the curve
            
        Returns:
            (coverage_values, accuracy_values)
        """
        if len(self.all_confidences) == 0:
            raise RuntimeError("No data to compute curve from")
        
        confidences = np.concatenate(self.all_confidences)
        predictions = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_preds = predictions[sorted_indices]
        sorted_targets = targets[sorted_indices]
        
        # Compute accuracy at different coverage levels
        coverage_values = []
        accuracy_values = []
        
        total_samples = len(sorted_preds)
        
        for i in range(1, n_points + 1):
            n_samples = int((i / n_points) * total_samples)
            if n_samples == 0:
                continue
            
            coverage = n_samples / total_samples
            accuracy = np.mean(sorted_preds[:n_samples] == sorted_targets[:n_samples])
            
            coverage_values.append(coverage)
            accuracy_values.append(accuracy)
        
        return np.array(coverage_values), np.array(accuracy_values)
    
    def plot_coverage_accuracy_curve(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot coverage vs accuracy curve.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        coverage, accuracy = self.compute_coverage_accuracy_curve()
        
        plt.figure(figsize=(10, 6))
        plt.plot(coverage, accuracy, marker='o', linewidth=2)
        plt.xlabel('Coverage (1 - Abstention Rate)')
        plt.ylabel('Accuracy')
        plt.title('Coverage vs Accuracy Trade-off')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    print("Testing OOD Metrics:")
    
    # Create dummy scores and labels
    np.random.seed(42)
    in_dist_scores = np.random.randn(500) + 2  # Higher scores
    ood_scores = np.random.randn(500) - 1      # Lower scores
    
    scores = np.concatenate([in_dist_scores, ood_scores])
    labels = np.concatenate([np.ones(500), np.zeros(500)])
    
    # Compute OOD metrics
    ood_metrics = OODMetrics()
    ood_metrics.update(scores, labels)
    results = ood_metrics.compute()
    
    print("\nOOD Detection Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Test threshold computation
    threshold = compute_likelihood_threshold(in_dist_scores, fpr_target=0.05)
    print(f"\nLikelihood threshold (5% FPR): {threshold:.4f}")
    
    # Test coverage-accuracy metrics
    print("\nTesting Coverage-Accuracy Metrics:")
    confidences = np.abs(np.random.randn(1000))
    predictions = np.random.randint(0, 2, 1000)
    targets = np.random.randint(0, 2, 1000)
    
    ca_metrics = CoverageAccuracyMetrics()
    ca_metrics.update(confidences, predictions, targets)
    coverage, accuracy = ca_metrics.compute_coverage_accuracy_curve()
    
    print(f"Coverage values: {coverage[:5]}")
    print(f"Accuracy values: {accuracy[:5]}")

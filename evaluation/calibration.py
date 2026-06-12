"""
Calibration Metrics

Evaluates model calibration and reliability:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagrams
- Brier score
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


class CalibrationMetrics:
    """
    Metrics for evaluating model calibration.
    
    Calibration measures how well predicted probabilities match
    actual frequencies of outcomes.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins for calibration computation
        """
        self.n_bins = n_bins
        self.reset()
    
    def reset(self):
        """Reset accumulated data."""
        self.all_probs = []
        self.all_targets = []
    
    def update(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray
    ):
        """
        Update with probabilities and targets.
        
        Args:
            probabilities: Predicted probabilities
            targets: Ground truth binary labels
        """
        if torch.is_tensor(probabilities):
            probabilities = probabilities.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        self.all_probs.append(probabilities.flatten())
        self.all_targets.append(targets.flatten())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute calibration metrics.
        
        Returns:
            Dictionary with ECE, MCE, and Brier score
        """
        if len(self.all_probs) == 0:
            raise RuntimeError("No data to compute metrics from")
        
        probs = np.concatenate(self.all_probs)
        targets = np.concatenate(self.all_targets)
        
        # Expected Calibration Error
        ece = self._compute_ece(probs, targets)
        
        # Maximum Calibration Error
        mce = self._compute_mce(probs, targets)
        
        # Brier Score
        brier = np.mean((probs - targets) ** 2)
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier)
        }
    
    def _compute_ece(
        self,
        probs: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE is the weighted average of calibration errors across bins.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Average confidence in bin
                avg_confidence = np.mean(probs[in_bin])
                
                # Average accuracy in bin
                avg_accuracy = np.mean(targets[in_bin])
                
                # Weighted calibration error
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece
    
    def _compute_mce(
        self,
        probs: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute Maximum Calibration Error.
        
        MCE is the maximum calibration error across all bins.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
            if np.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = np.mean(probs[in_bin])
                
                # Average accuracy in bin
                avg_accuracy = np.mean(targets[in_bin])
                
                # Calibration error
                error = np.abs(avg_confidence - avg_accuracy)
                mce = max(mce, error)
        
        return mce
    
    def get_reliability_diagram_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for reliability diagram.
        
        Returns:
            (bin_centers, accuracies, confidences)
        """
        if len(self.all_probs) == 0:
            raise RuntimeError("No data to compute diagram from")
        
        probs = np.concatenate(self.all_probs)
        targets = np.concatenate(self.all_targets)
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        accuracies = []
        confidences = []
        
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(probs[in_bin])
                avg_accuracy = np.mean(targets[in_bin])
                
                confidences.append(avg_confidence)
                accuracies.append(avg_accuracy)
            else:
                confidences.append(bin_centers[i])
                accuracies.append(0.0)
        
        return bin_centers, np.array(accuracies), np.array(confidences)
    
    def plot_reliability_diagram(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot reliability diagram.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        bin_centers, accuracies, confidences = self.get_reliability_diagram_data()
        
        plt.figure(figsize=(8, 8))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot actual calibration
        plt.plot(confidences, accuracies, 'o-', label='Model', linewidth=2, markersize=8)
        
        # Fill gap between perfect and actual
        plt.fill_between(confidences, accuracies, confidences, alpha=0.2)
        
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Reliability Diagram', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class RiskCoverageMetrics:
    """
    Risk-Coverage trade-off metrics.
    
    Evaluates how risk (error rate) changes with coverage.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset data."""
        self.all_confidences = []
        self.all_correct = []
    
    def update(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """
        Update with confidences, predictions, and targets.
        
        Args:
            confidences: Confidence scores
            predictions: Model predictions
            targets: Ground truth labels
        """
        if torch.is_tensor(confidences):
            confidences = confidences.detach().cpu().numpy()
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        correct = (predictions == targets).astype(np.float32)
        
        self.all_confidences.append(confidences.flatten())
        self.all_correct.append(correct.flatten())
    
    def compute_risk_coverage_curve(
        self,
        n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute risk vs coverage curve.
        
        Args:
            n_points: Number of points on curve
            
        Returns:
            (coverage_values, risk_values)
        """
        if len(self.all_confidences) == 0:
            raise RuntimeError("No data to compute curve from")
        
        confidences = np.concatenate(self.all_confidences)
        correct = np.concatenate(self.all_correct)
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_correct = correct[sorted_indices]
        
        coverage_values = []
        risk_values = []
        
        total_samples = len(sorted_correct)
        
        for i in range(1, n_points + 1):
            n_samples = int((i / n_points) * total_samples)
            if n_samples == 0:
                continue
            
            coverage = n_samples / total_samples
            risk = 1.0 - np.mean(sorted_correct[:n_samples])  # Error rate
            
            coverage_values.append(coverage)
            risk_values.append(risk)
        
        return np.array(coverage_values), np.array(risk_values)
    
    def plot_risk_coverage_curve(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot risk vs coverage curve.
        
        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        coverage, risk = self.compute_risk_coverage_curve()
        
        plt.figure(figsize=(10, 6))
        plt.plot(coverage, risk, marker='o', linewidth=2)
        plt.xlabel('Coverage (1 - Abstention Rate)')
        plt.ylabel('Risk (Error Rate)')
        plt.title('Risk-Coverage Trade-off')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    print("Testing Calibration Metrics:")
    
    # Create dummy probabilities and targets
    np.random.seed(42)
    
    # Well-calibrated predictions
    probs = np.random.rand(1000)
    targets = (np.random.rand(1000) < probs).astype(np.float32)
    
    # Compute calibration metrics
    cal_metrics = CalibrationMetrics(n_bins=10)
    cal_metrics.update(probs, targets)
    results = cal_metrics.compute()
    
    print("\nCalibration Metrics (well-calibrated):")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Overconfident predictions
    print("\nCalibration Metrics (overconfident):")
    overconfident_probs = np.minimum(probs * 1.5, 1.0)  # Boost probabilities
    
    cal_metrics2 = CalibrationMetrics(n_bins=10)
    cal_metrics2.update(overconfident_probs, targets)
    results2 = cal_metrics2.compute()
    
    for key, value in results2.items():
        print(f"  {key}: {value:.4f}")
    
    # Test risk-coverage
    print("\nTesting Risk-Coverage Metrics:")
    confidences = np.abs(np.random.randn(500))
    predictions = np.random.randint(0, 2, 500)
    targets_rc = np.random.randint(0, 2, 500)
    
    rc_metrics = RiskCoverageMetrics()
    rc_metrics.update(confidences, predictions, targets_rc)
    coverage, risk = rc_metrics.compute_risk_coverage_curve()
    
    print(f"Coverage values: {coverage[:5]}")
    print(f"Risk values: {risk[:5]}")

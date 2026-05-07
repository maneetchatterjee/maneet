"""
Perception Validation and Benchmarking Module

Quantifies perception quality with:
- Pose estimation error metrics
- Detection confidence vs success correlation
- Controlled noise experiments
- Ablation studies
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json

from ..perception.detector import PerceptionModule, Object3D


@dataclass
class PerceptionMetrics:
    """Metrics for perception quality."""
    detection_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    avg_position_error: float
    avg_orientation_error: float
    avg_confidence: float
    detection_rate: float  # TP / (TP + FN)
    precision: float  # TP / (TP + FP)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "detection_count": self.detection_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "avg_position_error": self.avg_position_error,
            "avg_orientation_error": self.avg_orientation_error,
            "avg_confidence": self.avg_confidence,
            "detection_rate": self.detection_rate,
            "precision": self.precision,
        }


class PerceptionValidator:
    """
    Validates perception module performance.
    
    Measures:
    - Pose estimation accuracy
    - Detection robustness to noise
    - Confidence calibration
    """
    
    def __init__(self, perception_module: PerceptionModule):
        """
        Initialize perception validator.
        
        Args:
            perception_module: Perception module to validate
        """
        self.perception = perception_module
        self.experiment_results = []
    
    def validate_with_ground_truth(
        self,
        rgb_image: np.ndarray,
        ground_truth_objects: List[Object3D],
        depth_image: Optional[np.ndarray] = None,
        camera_params: Optional[Dict] = None
    ) -> PerceptionMetrics:
        """
        Validate perception against ground truth.
        
        Args:
            rgb_image: Input RGB image
            ground_truth_objects: True object list
            depth_image: Optional depth image
            camera_params: Optional camera parameters
            
        Returns:
            PerceptionMetrics with accuracy statistics
        """
        # Run detection
        detected_objects = self.perception.detect_objects(
            rgb_image, depth_image, camera_params
        )
        
        # Match detections to ground truth
        matches, unmatched_detections, unmatched_gt = self._match_detections(
            detected_objects, ground_truth_objects
        )
        
        # Compute metrics
        true_positives = len(matches)
        false_positives = len(unmatched_detections)
        false_negatives = len(unmatched_gt)
        
        # Position errors
        position_errors = []
        orientation_errors = []
        confidences = []
        
        for detected, gt in matches:
            # Position error (Euclidean distance)
            pos_error = np.linalg.norm(
                np.array(detected.position) - np.array(gt.position)
            )
            position_errors.append(pos_error)
            
            # Orientation error (quaternion distance - simplified)
            ori_error = self._quaternion_distance(
                detected.orientation, gt.orientation
            )
            orientation_errors.append(ori_error)
            
            confidences.append(detected.confidence)
        
        # Aggregate metrics
        avg_pos_error = np.mean(position_errors) if position_errors else 0.0
        avg_ori_error = np.mean(orientation_errors) if orientation_errors else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        metrics = PerceptionMetrics(
            detection_count=len(detected_objects),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            avg_position_error=avg_pos_error,
            avg_orientation_error=avg_ori_error,
            avg_confidence=avg_confidence,
            detection_rate=detection_rate,
            precision=precision
        )
        
        return metrics
    
    def noise_robustness_experiment(
        self,
        clean_image: np.ndarray,
        ground_truth: List[Object3D],
        noise_levels: List[float] = None,
        num_trials: int = 10
    ) -> Dict[float, PerceptionMetrics]:
        """
        Test perception robustness to noise.
        
        Args:
            clean_image: Clean RGB image
            ground_truth: True object positions
            noise_levels: List of noise standard deviations
            num_trials: Number of trials per noise level
            
        Returns:
            Dictionary mapping noise level to metrics
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
        
        results = {}
        
        for noise_level in noise_levels:
            trial_metrics = []
            
            for trial in range(num_trials):
                # Add Gaussian noise
                noisy_image = self._add_noise(clean_image, noise_level)
                
                # Validate
                metrics = self.validate_with_ground_truth(
                    noisy_image, ground_truth
                )
                
                trial_metrics.append(metrics)
            
            # Average metrics across trials
            avg_metrics = self._average_metrics(trial_metrics)
            results[noise_level] = avg_metrics
        
        # Store for later analysis
        self.experiment_results.append({
            "experiment": "noise_robustness",
            "results": {k: v.to_dict() for k, v in results.items()}
        })
        
        return results
    
    def lighting_variation_experiment(
        self,
        image: np.ndarray,
        ground_truth: List[Object3D],
        brightness_factors: List[float] = None
    ) -> Dict[float, PerceptionMetrics]:
        """
        Test perception robustness to lighting changes.
        
        Args:
            image: Original RGB image
            ground_truth: True object positions
            brightness_factors: Brightness multiplication factors
            
        Returns:
            Dictionary mapping brightness to metrics
        """
        if brightness_factors is None:
            brightness_factors = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8]
        
        results = {}
        
        for factor in brightness_factors:
            # Adjust brightness
            adjusted_image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
            # Validate
            metrics = self.validate_with_ground_truth(
                adjusted_image, ground_truth
            )
            
            results[factor] = metrics
        
        # Store results
        self.experiment_results.append({
            "experiment": "lighting_variation",
            "results": {k: v.to_dict() for k, v in results.items()}
        })
        
        return results
    
    def occlusion_experiment(
        self,
        image: np.ndarray,
        ground_truth: List[Object3D],
        occlusion_ratios: List[float] = None
    ) -> Dict[float, PerceptionMetrics]:
        """
        Test perception robustness to occlusions.
        
        Args:
            image: RGB image
            ground_truth: True objects
            occlusion_ratios: Fraction of image to occlude
            
        Returns:
            Dictionary mapping occlusion ratio to metrics
        """
        if occlusion_ratios is None:
            occlusion_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        results = {}
        
        for ratio in occlusion_ratios:
            # Add random occlusions
            occluded_image = self._add_occlusions(image, ratio)
            
            # Validate
            metrics = self.validate_with_ground_truth(
                occluded_image, ground_truth
            )
            
            results[ratio] = metrics
        
        # Store results
        self.experiment_results.append({
            "experiment": "occlusion",
            "results": {k: v.to_dict() for k, v in results.items()}
        })
        
        return results
    
    def confidence_calibration_analysis(
        self,
        test_images: List[np.ndarray],
        ground_truths: List[List[Object3D]]
    ) -> Dict:
        """
        Analyze confidence calibration (confidence vs accuracy).
        
        Args:
            test_images: List of test images
            ground_truths: Corresponding ground truth objects
            
        Returns:
            Calibration statistics
        """
        all_confidences = []
        all_errors = []
        
        for img, gt in zip(test_images, ground_truths):
            detected = self.perception.detect_objects(img)
            matches, _, _ = self._match_detections(detected, gt)
            
            for det, gt_obj in matches:
                pos_error = np.linalg.norm(
                    np.array(det.position) - np.array(gt_obj.position)
                )
                all_confidences.append(det.confidence)
                all_errors.append(pos_error)
        
        # Compute correlation
        if all_confidences and all_errors:
            correlation = np.corrcoef(all_confidences, all_errors)[0, 1]
        else:
            correlation = 0.0
        
        # Bin by confidence
        confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        binned_errors = {f"{confidence_bins[i]}-{confidence_bins[i+1]}": [] 
                        for i in range(len(confidence_bins)-1)}
        
        for conf, err in zip(all_confidences, all_errors):
            for i in range(len(confidence_bins)-1):
                if confidence_bins[i] <= conf < confidence_bins[i+1]:
                    key = f"{confidence_bins[i]}-{confidence_bins[i+1]}"
                    binned_errors[key].append(err)
                    break
        
        avg_errors_by_conf = {
            k: np.mean(v) if v else 0.0 
            for k, v in binned_errors.items()
        }
        
        return {
            "confidence_error_correlation": correlation,
            "avg_errors_by_confidence": avg_errors_by_conf,
            "total_samples": len(all_confidences)
        }
    
    def _match_detections(
        self,
        detections: List[Object3D],
        ground_truth: List[Object3D],
        position_threshold: float = 0.1
    ) -> Tuple[List[Tuple[Object3D, Object3D]], List[Object3D], List[Object3D]]:
        """
        Match detections to ground truth.
        
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_ground_truth)
        """
        matches = []
        unmatched_det = list(detections)
        unmatched_gt = list(ground_truth)
        
        # Simple greedy matching based on position
        for gt in ground_truth:
            best_match = None
            best_distance = float('inf')
            
            for det in unmatched_det:
                # Check if color and shape match
                if det.color != gt.color or det.shape != gt.shape:
                    continue
                
                # Check position distance
                distance = np.linalg.norm(
                    np.array(det.position) - np.array(gt.position)
                )
                
                if distance < position_threshold and distance < best_distance:
                    best_match = det
                    best_distance = distance
            
            if best_match:
                matches.append((best_match, gt))
                unmatched_det.remove(best_match)
                unmatched_gt.remove(gt)
        
        return matches, unmatched_det, unmatched_gt
    
    def _quaternion_distance(
        self,
        q1: Tuple[float, float, float, float],
        q2: Tuple[float, float, float, float]
    ) -> float:
        """Compute distance between two quaternions."""
        # Simplified - just use dot product
        dot = sum(a * b for a, b in zip(q1, q2))
        return 1.0 - abs(dot)  # Distance in [0, 1]
    
    def _add_noise(self, image: np.ndarray, noise_std: float) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, noise_std * 255, image.shape)
        noisy = image.astype(float) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def _add_occlusions(self, image: np.ndarray, ratio: float) -> np.ndarray:
        """Add random rectangular occlusions."""
        occluded = image.copy()
        h, w = image.shape[:2]
        
        # Number of occlusion patches
        num_patches = int(ratio * 10)
        
        for _ in range(num_patches):
            # Random patch size and position
            patch_h = np.random.randint(20, 100)
            patch_w = np.random.randint(20, 100)
            y = np.random.randint(0, max(1, h - patch_h))
            x = np.random.randint(0, max(1, w - patch_w))
            
            # Black occlusion
            occluded[y:y+patch_h, x:x+patch_w] = 0
        
        return occluded
    
    def _average_metrics(self, metrics_list: List[PerceptionMetrics]) -> PerceptionMetrics:
        """Average multiple perception metrics."""
        return PerceptionMetrics(
            detection_count=int(np.mean([m.detection_count for m in metrics_list])),
            true_positives=int(np.mean([m.true_positives for m in metrics_list])),
            false_positives=int(np.mean([m.false_positives for m in metrics_list])),
            false_negatives=int(np.mean([m.false_negatives for m in metrics_list])),
            avg_position_error=np.mean([m.avg_position_error for m in metrics_list]),
            avg_orientation_error=np.mean([m.avg_orientation_error for m in metrics_list]),
            avg_confidence=np.mean([m.avg_confidence for m in metrics_list]),
            detection_rate=np.mean([m.detection_rate for m in metrics_list]),
            precision=np.mean([m.precision for m in metrics_list]),
        )
    
    def export_results(self, filepath: str):
        """Export all experiment results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
    
    def generate_summary_report(self) -> str:
        """Generate text summary of validation results."""
        if not self.experiment_results:
            return "No experiments run yet."
        
        report = ["="*60, "PERCEPTION VALIDATION REPORT", "="*60, ""]
        
        for exp in self.experiment_results:
            report.append(f"\nExperiment: {exp['experiment']}")
            report.append("-"*40)
            
            for condition, metrics in exp['results'].items():
                if isinstance(metrics, dict):
                    report.append(f"\n  Condition: {condition}")
                    report.append(f"    Detection Rate: {metrics.get('detection_rate', 0):.3f}")
                    report.append(f"    Precision: {metrics.get('precision', 0):.3f}")
                    report.append(f"    Avg Position Error: {metrics.get('avg_position_error', 0):.4f}m")
        
        report.append("\n" + "="*60)
        return "\n".join(report)

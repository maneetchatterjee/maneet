"""Perception module for object detection."""

from .detector import PerceptionModule, Object3D, ShapeDetector
from .validation import PerceptionValidator, PerceptionMetrics
from .formal_verification import (
    ComprehensivePerceptionVerification,
    DatasetDescription,
    SampleSizeJustification,
    ConfidenceIntervalAnalysis,
    HypothesisTestingFramework,
    AdversarialTests,
    PrecisionRecallAnalysis,
    FailureModeClustering
)

__all__ = [
    "PerceptionModule", 
    "Object3D", 
    "ShapeDetector",
    "PerceptionValidator",
    "PerceptionMetrics",
    "ComprehensivePerceptionVerification",
    "DatasetDescription",
    "SampleSizeJustification",
    "ConfidenceIntervalAnalysis",
    "HypothesisTestingFramework",
    "AdversarialTests",
    "PrecisionRecallAnalysis",
    "FailureModeClustering"
]

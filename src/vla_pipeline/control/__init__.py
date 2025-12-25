"""Control module for kinematics and trajectory execution."""

from .kinematics import KinematicsController, TrajectoryExecutor
from .enhanced_kinematics import (
    EnhancedKinematicsController,
    TrajectoryExecutor as EnhancedTrajectoryExecutor,
    IKMetrics
)
from .formal_verification import (
    ComprehensiveIKVerification,
    MathematicalDerivation,
    StabilityAnalysis,
    DampingCoefficientJustification,
    JointLimitEnforcementProof,
    RandomizedWorkspaceTests,
    SingularConfigurationTests,
    BaselineComparison,
    IKVerificationResult,
    export_verification_report
)

__all__ = [
    "KinematicsController", 
    "TrajectoryExecutor",
    "EnhancedKinematicsController",
    "EnhancedTrajectoryExecutor",
    "IKMetrics",
    # Verification framework
    "ComprehensiveIKVerification",
    "MathematicalDerivation",
    "StabilityAnalysis",
    "DampingCoefficientJustification",
    "JointLimitEnforcementProof",
    "RandomizedWorkspaceTests",
    "SingularConfigurationTests",
    "BaselineComparison",
    "IKVerificationResult",
    "export_verification_report"
]

#!/usr/bin/env python3
"""
Test suite for IK formal verification.

Runs comprehensive verification of the DLS IK solver and generates report.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vla_pipeline.control.enhanced_kinematics import EnhancedKinematicsController
from src.vla_pipeline.control.formal_verification import (
    ComprehensiveIKVerification,
    export_verification_report
)


def main():
    """Run comprehensive IK verification."""
    print("Initializing IK controller...")
    controller = EnhancedKinematicsController(
        damping_factor=0.01,
        singularity_threshold=0.001
    )
    
    print("Running comprehensive verification suite...")
    print("(This may take 1-2 minutes)\n")
    
    report = ComprehensiveIKVerification.run_full_verification(
        controller,
        num_workspace_samples=500,
        num_lambda_tests=10,
        verbose=True
    )
    
    # Export report
    export_verification_report(report, "ik_verification_report.json")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"\nFull report exported to: ik_verification_report.json")
    print("Run generate_ik_plots.py to create visualization plots.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

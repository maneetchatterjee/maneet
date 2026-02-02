#!/usr/bin/env python3
"""
Research-Grade VLA Pipeline Demo

Demonstrates enhanced features:
1. Semantic parsing vs rule-based
2. Symbolic planning with state tracking
3. Enhanced IK with singularity handling
4. Perception validation with noise experiments
5. Comprehensive benchmarking and ablation studies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vla_pipeline.language import SemanticParser, LanguageReasoningModule
from vla_pipeline.planning import StateBasedPlanner, PlanningModule
from vla_pipeline.control import EnhancedKinematicsController
from vla_pipeline.perception import PerceptionModule, PerceptionValidator, Object3D
from vla_pipeline.utils import AblationStudy


def demo_semantic_parsing():
    """Demonstrate enhanced semantic parsing."""
    print("\n" + "="*70)
    print("DEMO 1: Semantic Parsing (Research-Grade)")
    print("="*70)
    
    parser = SemanticParser()
    
    test_commands = [
        "Pick the red cube and place it left of the blue cube",
        "Grab the small green sphere and put it on the yellow block",
        "Stack the red box on the blue box",
    ]
    
    for cmd in test_commands:
        print(f"\nCommand: '{cmd}'")
        program = parser.parse(cmd)
        print(f"Semantic Program:")
        print(program.to_json())
        
        # Validate
        is_valid, errors = parser.validate_program(program)
        print(f"Valid: {is_valid}")
        if errors:
            print(f"Errors: {errors}")


def demo_symbolic_planning():
    """Demonstrate STRIPS-style symbolic planning."""
    print("\n" + "="*70)
    print("DEMO 2: Symbolic Planning with State Tracking")
    print("="*70)
    
    planner = StateBasedPlanner()
    
    # Create test objects
    objects = [
        Object3D(0, "red_cube", "red", "cube", (0.3, 0.0, 0.05), (0,0,0,1), (0.05,0.05,0.05)),
        Object3D(1, "blue_cube", "blue", "cube", (0.3, 0.15, 0.05), (0,0,0,1), (0.05,0.05,0.05)),
    ]
    
    # Initialize state
    state = planner.initialize_state(objects)
    print("\nInitial State Predicates:")
    for pred in list(state.predicates)[:10]:  # Show first 10
        print(f"  {pred}")
    
    # Parse command
    parser = SemanticParser()
    semantic_prog = parser.parse("Pick the red cube and place it left of the blue cube")
    
    # Plan
    print("\nPlanning...")
    plan = planner.plan(semantic_prog, state)
    
    if plan:
        print(f"\nGenerated Plan ({len(plan)} actions):")
        for i, action in enumerate(plan):
            print(f"  {i+1}. {action.name}({action.parameters})")
            print(f"     Preconditions: {[str(p) for p in action.preconditions][:3]}")
            print(f"     Effects: +{[str(p) for p in action.add_effects][:2]}")
    else:
        print("\nPlanning failed!")


def demo_enhanced_ik():
    """Demonstrate enhanced IK with singularity handling."""
    print("\n" + "="*70)
    print("DEMO 3: Enhanced IK with Damped Least Squares")
    print("="*70)
    
    controller = EnhancedKinematicsController(
        damping_factor=0.01,
        singularity_threshold=0.001
    )
    
    # Test poses including near-singularity
    test_poses = [
        ((0.3, 0.0, 0.1), (0,0,0,1), "Normal pose"),
        ((0.0, 0.0, 0.1), (0,0,0,1), "Near singularity (origin)"),
        ((0.5, 0.0, 0.1), (0,0,0,1), "Workspace boundary"),
    ]
    
    for pos, orn, description in test_poses:
        print(f"\n{description}: {pos}")
        joints, metrics = controller.inverse_kinematics(pos, orn)
        
        print(f"  Converged: {metrics.converged}")
        print(f"  Iterations: {metrics.iterations}")
        print(f"  Final Error: {metrics.final_error:.6f}m")
        print(f"  Singularity: {metrics.singularity_encountered}")
        print(f"  Joint Violations: {metrics.joint_limit_violations}")
        print(f"  Solution: {joints[:3]}")  # Show first 3 joints


def demo_perception_validation():
    """Demonstrate perception validation with noise experiments."""
    print("\n" + "="*70)
    print("DEMO 4: Perception Validation and Noise Robustness")
    print("="*70)
    
    perception = PerceptionModule()
    validator = PerceptionValidator(perception)
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:200, 100:200] = [255, 0, 0]  # Red
    test_image[300:400, 300:400] = [0, 0, 255]  # Blue
    
    # Ground truth
    ground_truth = [
        Object3D(0, "red_cube", "red", "cube", (0.15, 0.15, 0), (0,0,0,1), (0.05,0.05,0.05)),
        Object3D(1, "blue_cube", "blue", "cube", (0.35, 0.35, 0), (0,0,0,1), (0.05,0.05,0.05)),
    ]
    
    # Run noise robustness experiment
    print("\nRunning noise robustness experiment...")
    noise_results = validator.noise_robustness_experiment(
        test_image, ground_truth,
        noise_levels=[0.0, 0.05, 0.1, 0.2],
        num_trials=3
    )
    
    print("\nNoise Robustness Results:")
    print(f"{'Noise Level':<15} {'Detection Rate':<18} {'Precision':<12}")
    print("-"*45)
    for noise, metrics in noise_results.items():
        print(f"{noise:<15.2f} {metrics.detection_rate:<18.3f} {metrics.precision:<12.3f}")


def demo_ablation_study():
    """Demonstrate ablation study comparing methods."""
    print("\n" + "="*70)
    print("DEMO 5: Ablation Study - Method Comparison")
    print("="*70)
    
    ablation = AblationStudy()
    
    # Test commands
    test_commands = [
        "Pick the red cube",
        "Place the blue sphere on the table",
        "Move the green block to the left",
        "Grab the yellow cylinder and put it next to the red cube",
    ]
    
    # Parsers
    rule_based = LanguageReasoningModule()
    semantic = SemanticParser()
    
    print("\nComparing Language Parsing Methods...")
    comparison = ablation.compare_language_parsing(
        test_commands,
        rule_based,
        semantic
    )
    
    print(f"\nLanguage Parsing Comparison:")
    print(f"  Rule-Based:")
    print(f"    Success Rate: {comparison['rule_based']['success_rate']:.1%}")
    print(f"    Avg Time: {comparison['rule_based']['avg_time_ms']:.2f}ms")
    
    print(f"  Semantic:")
    print(f"    Success Rate: {comparison['semantic']['success_rate']:.1%}")
    print(f"    Avg Time: {comparison['semantic']['avg_time_ms']:.2f}ms")
    
    # Generate LaTeX table
    print("\nLaTeX Table:")
    print(ablation.generate_latex_table([comparison]))


def demo_comprehensive_metrics():
    """Show comprehensive performance metrics."""
    print("\n" + "="*70)
    print("DEMO 6: Comprehensive Performance Metrics")
    print("="*70)
    
    # IK Performance
    controller = EnhancedKinematicsController()
    
    print("\nIK Performance on 10 Random Poses:")
    metrics_list = []
    for i in range(10):
        pos = (np.random.uniform(-0.3, 0.3), 
               np.random.uniform(-0.3, 0.3), 
               np.random.uniform(0.05, 0.3))
        orn = (0, 0, 0, 1)
        
        joints, metrics = controller.inverse_kinematics(pos, orn)
        metrics_list.append(metrics)
    
    summary = controller.get_ik_performance_summary(metrics_list)
    
    print(f"  Total IK Calls: {summary['total_ik_calls']}")
    print(f"  Convergence Rate: {summary['convergence_rate']:.1%}")
    print(f"  Singularity Rate: {summary['singularity_rate']:.1%}")
    print(f"  Avg Iterations: {summary['avg_iterations']:.1f}")
    print(f"  Avg Final Error: {summary['avg_final_error']:.6f}m")
    print(f"  Max Final Error: {summary['max_final_error']:.6f}m")
    print(f"  Total Joint Violations: {summary['total_joint_violations']}")


def main():
    """Run all research-grade demos."""
    print("="*70)
    print("RESEARCH-GRADE VLA PIPELINE DEMONSTRATION")
    print("Showcasing Enhanced Features for Publication-Quality Results")
    print("="*70)
    
    demos = [
        ("Semantic Parsing", demo_semantic_parsing),
        ("Symbolic Planning", demo_symbolic_planning),
        ("Enhanced IK", demo_enhanced_ik),
        ("Perception Validation", demo_perception_validation),
        ("Ablation Study", demo_ablation_study),
        ("Comprehensive Metrics", demo_comprehensive_metrics),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Compositional semantic parsing")
    print("  ✓ STRIPS-style symbolic planning with state tracking")
    print("  ✓ Damped least squares IK with singularity handling")
    print("  ✓ Perception validation with noise experiments")
    print("  ✓ Ablation studies and method comparisons")
    print("  ✓ Comprehensive performance metrics and reporting")
    print("\nFor full documentation, see docs/RESEARCH_SPECIFICATION.md")
    print("="*70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for planner formal verification.

Runs comprehensive verification tests and generates report.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vla_pipeline.planning.symbolic_planner import StateBasedPlanner
from vla_pipeline.planning.formal_verification import (
    run_comprehensive_verification
)
import json


def main():
    """Run verification tests."""
    print("=" * 80)
    print("PLANNER FORMAL VERIFICATION")
    print("=" * 80)
    print()
    
    # Create planner
    planner = StateBasedPlanner()
    
    # Run comprehensive verification
    results = run_comprehensive_verification(planner)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    print("\n1. Soundness:")
    print(f"   Pass rate: {results['soundness']['summary']['pass_rate']:.1%}")
    print(f"   Tests: {results['soundness']['summary']['passed']}/{results['soundness']['summary']['total_tests']}")
    print(f"   Violations: {len(results['soundness']['summary']['violations'])}")
    
    print("\n2. Completeness:")
    print(f"   Success rate: {results['completeness']['summary']['success_rate']:.1%}")
    print(f"   Found plans: {results['completeness']['summary']['found_plans']}/{results['completeness']['summary']['total_tests']}")
    print(f"   False negatives: {results['completeness']['summary']['missed_plans']}")
    
    print("\n3. Complexity:")
    print(f"   State space size: {results['complexity']['state_space_size']:,}")
    print(f"   Estimated time complexity: O({results['complexity']['estimated_time_complexity']:,})")
    print(f"   Avg empirical time: {results['complexity']['empirical']['avg_time']:.4f}s")
    print(f"   Avg plan length: {results['complexity']['empirical']['avg_plan_length']:.1f}")
    
    print("\n4. Failure-Inducing Worlds:")
    for world_type, result in results['failure_worlds'].items():
        status = "✓ SOLVED" if result['solvable'] else "✗ UNSOLVED"
        length = f" (length {result['plan_length']})" if result['plan_length'] else ""
        print(f"   {world_type:20s}: {status}{length}")
    
    print("\n5. Replanning Termination:")
    term_result = results['replanning']['termination_test']
    print(f"   Terminated: {term_result['terminated']}")
    print(f"   Reason: {term_result['reason']}")
    print(f"   Replan count: {term_result['replan_count']}")
    print(f"   Infinite loops detected: {results['replanning']['infinite_loops']}")
    
    print("\n6. Baseline Comparison:")
    comp = results['baseline_comparison']
    print(f"   STRIPS success rate: {comp['strips']['success_rate']:.1%}")
    print(f"   Greedy success rate: {comp['greedy']['success_rate']:.1%}")
    print(f"   Random success rate: {comp['random']['success_rate']:.1%}")
    print(f"   Winner: {comp['comparison']['winner']}")
    
    # Save to JSON
    output_file = '/home/runner/work/maneet/maneet/planner_verification_report.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFull report saved to: {output_file}")
    print("\n" + "=" * 80)
    
    # Determine overall verdict
    soundness_ok = results['soundness']['summary']['pass_rate'] >= 0.9
    completeness_ok = results['completeness']['summary']['success_rate'] >= 0.8
    no_infinite_loops = results['replanning']['infinite_loops'] == 0
    
    if soundness_ok and completeness_ok and no_infinite_loops:
        print("VERDICT: ✓ VERIFIED (with documented limitations)")
    else:
        print("VERDICT: ⚠ PARTIALLY VERIFIED (critical issues found)")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

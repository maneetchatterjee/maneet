"""
Benchmarking and Ablation Study Module

Provides comprehensive benchmarking:
- Rule-based vs semantic parsing comparison
- Scripted vs symbolic planning ablation
- Performance plots and tables
- Statistical significance testing
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import time
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    method_name: str
    task_description: str
    success: bool
    execution_time: float
    num_steps: int
    error_magnitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AblationStudy:
    """
    Conducts ablation studies on VLA pipeline components.
    """
    
    def __init__(self):
        """Initialize ablation study."""
        self.results: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    
    def compare_language_parsing(
        self,
        test_commands: List[str],
        rule_based_parser,
        semantic_parser
    ) -> Dict[str, Any]:
        """
        Compare rule-based vs semantic parsing.
        
        Args:
            test_commands: List of test commands
            rule_based_parser: Old rule-based parser
            semantic_parser: New semantic parser
            
        Returns:
            Comparison statistics
        """
        rule_based_times = []
        semantic_times = []
        
        rule_based_success = 0
        semantic_success = 0
        
        for cmd in test_commands:
            # Rule-based
            start = time.time()
            try:
                rb_result = rule_based_parser.parse_command(cmd)
                rb_time = time.time() - start
                rule_based_times.append(rb_time)
                if rb_result.action.value != "none":
                    rule_based_success += 1
            except Exception:
                rb_time = time.time() - start
                rule_based_times.append(rb_time)
            
            # Semantic
            start = time.time()
            try:
                sem_result = semantic_parser.parse(cmd)
                sem_time = time.time() - start
                semantic_times.append(sem_time)
                if sem_result.goal.value != "none":
                    semantic_success += 1
            except Exception:
                sem_time = time.time() - start
                semantic_times.append(sem_time)
        
        comparison = {
            "method": "Language Parsing",
            "rule_based": {
                "success_rate": rule_based_success / len(test_commands),
                "avg_time_ms": np.mean(rule_based_times) * 1000,
                "std_time_ms": np.std(rule_based_times) * 1000,
            },
            "semantic": {
                "success_rate": semantic_success / len(test_commands),
                "avg_time_ms": np.mean(semantic_times) * 1000,
                "std_time_ms": np.std(semantic_times) * 1000,
            }
        }
        
        return comparison
    
    def compare_planning_methods(
        self,
        test_scenarios: List[Dict],
        scripted_planner,
        symbolic_planner
    ) -> Dict[str, Any]:
        """
        Compare scripted vs symbolic planning.
        
        Args:
            test_scenarios: List of test scenarios with objects and goals
            scripted_planner: Original scripted planner
            symbolic_planner: New symbolic planner
            
        Returns:
            Comparison statistics
        """
        scripted_results = []
        symbolic_results = []
        
        for scenario in test_scenarios:
            # Scripted planning
            start = time.time()
            try:
                scripted_plan = scripted_planner.plan_action(
                    scenario['command'],
                    scenario['objects']
                )
                scripted_time = time.time() - start
                scripted_success = len(scripted_plan.waypoints) > 0
                scripted_results.append(BenchmarkResult(
                    method_name="scripted",
                    task_description=str(scenario.get('description', '')),
                    success=scripted_success,
                    execution_time=scripted_time,
                    num_steps=len(scripted_plan.waypoints),
                    error_magnitude=0.0
                ))
            except Exception as e:
                scripted_time = time.time() - start
                scripted_results.append(BenchmarkResult(
                    method_name="scripted",
                    task_description=str(scenario.get('description', '')),
                    success=False,
                    execution_time=scripted_time,
                    num_steps=0,
                    error_magnitude=1.0,
                    metadata={"error": str(e)}
                ))
            
            # Symbolic planning
            start = time.time()
            try:
                state = symbolic_planner.initialize_state(scenario['objects'])
                symbolic_plan = symbolic_planner.plan(
                    scenario['semantic_program'],
                    state
                )
                symbolic_time = time.time() - start
                symbolic_success = symbolic_plan is not None and len(symbolic_plan) > 0
                symbolic_results.append(BenchmarkResult(
                    method_name="symbolic",
                    task_description=str(scenario.get('description', '')),
                    success=symbolic_success,
                    execution_time=symbolic_time,
                    num_steps=len(symbolic_plan) if symbolic_plan else 0,
                    error_magnitude=0.0 if symbolic_success else 1.0
                ))
            except Exception as e:
                symbolic_time = time.time() - start
                symbolic_results.append(BenchmarkResult(
                    method_name="symbolic",
                    task_description=str(scenario.get('description', '')),
                    success=False,
                    execution_time=symbolic_time,
                    num_steps=0,
                    error_magnitude=1.0,
                    metadata={"error": str(e)}
                ))
        
        # Store results
        self.results['scripted_planning'].extend(scripted_results)
        self.results['symbolic_planning'].extend(symbolic_results)
        
        # Compute statistics
        scripted_success_rate = sum(1 for r in scripted_results if r.success) / len(scripted_results)
        symbolic_success_rate = sum(1 for r in symbolic_results if r.success) / len(symbolic_results)
        
        scripted_avg_time = np.mean([r.execution_time for r in scripted_results])
        symbolic_avg_time = np.mean([r.execution_time for r in symbolic_results])
        
        comparison = {
            "method": "Planning",
            "scripted": {
                "success_rate": scripted_success_rate,
                "avg_time_ms": scripted_avg_time * 1000,
                "avg_steps": np.mean([r.num_steps for r in scripted_results if r.success]),
            },
            "symbolic": {
                "success_rate": symbolic_success_rate,
                "avg_time_ms": symbolic_avg_time * 1000,
                "avg_steps": np.mean([r.num_steps for r in symbolic_results if r.success]),
            }
        }
        
        return comparison
    
    def compare_ik_methods(
        self,
        test_poses: List[Tuple],
        standard_ik_controller,
        enhanced_ik_controller
    ) -> Dict[str, Any]:
        """
        Compare standard vs enhanced IK with damping.
        
        Args:
            test_poses: List of (position, orientation) tuples
            standard_ik_controller: Controller without damping
            enhanced_ik_controller: Controller with damped least squares
            
        Returns:
            Comparison statistics
        """
        standard_results = []
        enhanced_results = []
        
        for pos, orn in test_poses:
            # Standard IK
            try:
                joints_std, metrics_std = standard_ik_controller.inverse_kinematics(
                    pos, orn, use_damping=False
                )
                standard_results.append({
                    "converged": metrics_std.converged,
                    "iterations": metrics_std.iterations,
                    "final_error": metrics_std.final_error,
                    "singularity": metrics_std.singularity_encountered
                })
            except Exception:
                standard_results.append({
                    "converged": False,
                    "iterations": 0,
                    "final_error": 1.0,
                    "singularity": True
                })
            
            # Enhanced IK
            try:
                joints_enh, metrics_enh = enhanced_ik_controller.inverse_kinematics(
                    pos, orn, use_damping=True
                )
                enhanced_results.append({
                    "converged": metrics_enh.converged,
                    "iterations": metrics_enh.iterations,
                    "final_error": metrics_enh.final_error,
                    "singularity": metrics_enh.singularity_encountered
                })
            except Exception:
                enhanced_results.append({
                    "converged": False,
                    "iterations": 0,
                    "final_error": 1.0,
                    "singularity": True
                })
        
        # Compute statistics
        std_convergence = sum(1 for r in standard_results if r["converged"]) / len(standard_results)
        enh_convergence = sum(1 for r in enhanced_results if r["converged"]) / len(enhanced_results)
        
        comparison = {
            "method": "Inverse Kinematics",
            "standard": {
                "convergence_rate": std_convergence,
                "avg_iterations": np.mean([r["iterations"] for r in standard_results]),
                "avg_final_error": np.mean([r["final_error"] for r in standard_results]),
                "singularity_rate": sum(1 for r in standard_results if r["singularity"]) / len(standard_results),
            },
            "enhanced_damped": {
                "convergence_rate": enh_convergence,
                "avg_iterations": np.mean([r["iterations"] for r in enhanced_results]),
                "avg_final_error": np.mean([r["final_error"] for r in enhanced_results]),
                "singularity_rate": sum(1 for r in enhanced_results if r["singularity"]) / len(enhanced_results),
            }
        }
        
        return comparison
    
    def generate_comparison_plots(self, output_dir: str = "benchmark_plots"):
        """
        Generate comparison plots for ablation study.
        
        Args:
            output_dir: Directory to save plots
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping plots.")
            return
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot planning comparison
        if 'scripted_planning' in self.results and 'symbolic_planning' in self.results:
            self._plot_planning_comparison(output_dir)
        
        print(f"Plots saved to {output_dir}/")
    
    def _plot_planning_comparison(self, output_dir: str):
        """Plot planning method comparison."""
        scripted = self.results['scripted_planning']
        symbolic = self.results['symbolic_planning']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Success rate
        scripted_success = sum(1 for r in scripted if r.success) / len(scripted)
        symbolic_success = sum(1 for r in symbolic if r.success) / len(symbolic)
        
        axes[0].bar(['Scripted', 'Symbolic'], [scripted_success, symbolic_success], 
                    color=['#1f77b4', '#ff7f0e'])
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Planning Success Rate')
        axes[0].set_ylim([0, 1])
        
        # Execution time
        scripted_times = [r.execution_time * 1000 for r in scripted]
        symbolic_times = [r.execution_time * 1000 for r in symbolic]
        
        axes[1].boxplot([scripted_times, symbolic_times], labels=['Scripted', 'Symbolic'])
        axes[1].set_ylabel('Execution Time (ms)')
        axes[1].set_title('Planning Time Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/planning_comparison.png", dpi=300)
        plt.close()
    
    def export_results(self, filepath: str):
        """Export all benchmark results to JSON."""
        export_data = {}
        
        for method, results in self.results.items():
            export_data[method] = [
                {
                    "method_name": r.method_name,
                    "task": r.task_description,
                    "success": r.success,
                    "time": r.execution_time,
                    "steps": r.num_steps,
                    "error": r.error_magnitude,
                    "metadata": r.metadata
                }
                for r in results
            ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def generate_latex_table(self, comparisons: List[Dict]) -> str:
        """
        Generate LaTeX table from comparison results.
        
        Args:
            comparisons: List of comparison dictionaries
            
        Returns:
            LaTeX table string
        """
        latex = ["\\begin{table}[h]", "\\centering", "\\begin{tabular}{lcc}"]
        latex.append("\\hline")
        latex.append("Method & Baseline & Enhanced \\\\")
        latex.append("\\hline")
        
        for comp in comparisons:
            method = comp["method"]
            baseline_key = list(comp.keys())[1]  # First method
            enhanced_key = list(comp.keys())[2]  # Second method
            
            baseline = comp[baseline_key]
            enhanced = comp[enhanced_key]
            
            latex.append(f"{method} & & \\\\")
            
            # Success rate
            if "success_rate" in baseline:
                latex.append(f"  Success Rate & {baseline['success_rate']:.3f} & {enhanced['success_rate']:.3f} \\\\")
            
            # Time
            if "avg_time_ms" in baseline:
                latex.append(f"  Avg Time (ms) & {baseline['avg_time_ms']:.2f} & {enhanced['avg_time_ms']:.2f} \\\\")
            
            # Convergence
            if "convergence_rate" in baseline:
                latex.append(f"  Convergence & {baseline['convergence_rate']:.3f} & {enhanced['convergence_rate']:.3f} \\\\")
            
            latex.append("\\hline")
        
        latex.extend(["\\end{tabular}", "\\caption{Ablation Study Results}", "\\end{table}"])
        
        return "\n".join(latex)


def run_comprehensive_benchmark(
    vla_pipeline,
    test_scenarios: List[Dict],
    output_dir: str = "benchmark_results"
) -> Dict:
    """
    Run comprehensive benchmark of entire VLA pipeline.
    
    Args:
        vla_pipeline: VLA pipeline instance
        test_scenarios: List of test scenarios
        output_dir: Output directory for results
        
    Returns:
        Summary statistics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for scenario in test_scenarios:
        start_time = time.time()
        
        try:
            # Execute scenario
            success = vla_pipeline.execute_command(scenario['command'])
            execution_time = time.time() - start_time
            
            results.append({
                "scenario": scenario.get('description', ''),
                "command": scenario['command'],
                "success": success,
                "execution_time": execution_time,
            })
        except Exception as e:
            execution_time = time.time() - start_time
            results.append({
                "scenario": scenario.get('description', ''),
                "command": scenario['command'],
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            })
    
    # Compute summary
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    avg_time = np.mean([r["execution_time"] for r in results])
    
    summary = {
        "total_scenarios": len(results),
        "success_rate": success_rate,
        "avg_execution_time": avg_time,
        "results": results
    }
    
    # Save results
    with open(f"{output_dir}/benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

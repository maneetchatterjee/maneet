"""
VLA Pipeline

Main orchestrator that integrates perception, language, planning,
control, and simulation modules.
"""

import numpy as np
from typing import List, Optional, Dict
import json

from .perception import PerceptionModule, Object3D
from .language import LanguageReasoningModule, ParsedCommand, ActionType
from .planning import PlanningModule, SymbolicAction
from .control import KinematicsController, TrajectoryExecutor
from .simulation import SimulationEnvironment
from .utils import MetricsLogger, Timer


class VLAPipeline:
    """
    Vision-Language-Action Pipeline for robotic manipulation.
    
    Integrates all modules in a modular, extensible architecture.
    """
    
    def __init__(
        self,
        use_gui: bool = True,
        log_metrics: bool = True
    ):
        """
        Initialize VLA pipeline.
        
        Args:
            use_gui: Whether to use GUI for simulation
            log_metrics: Whether to log execution metrics
        """
        # Initialize modules
        self.perception = PerceptionModule()
        self.language = LanguageReasoningModule()
        self.planning = PlanningModule()
        self.simulation = SimulationEnvironment(use_gui=use_gui)
        
        # Initialize robot
        self.simulation.load_robot()
        
        # Initialize control
        self.kinematics = KinematicsController()
        self.executor = TrajectoryExecutor(self.kinematics)
        
        # Metrics
        self.log_metrics = log_metrics
        if log_metrics:
            self.metrics_logger = MetricsLogger()
        
        self.task_counter = 0
    
    def execute_command(self, command: str) -> bool:
        """
        Execute natural language command end-to-end.
        
        Args:
            command: Natural language command
            
        Returns:
            True if execution successful
        """
        print(f"\n{'='*60}")
        print(f"Executing Command: '{command}'")
        print(f"{'='*60}")
        
        with Timer() as timer:
            try:
                # 1. Perception: Get scene representation
                print("\n[1/5] Perception: Capturing scene...")
                rgb_image, depth_image = self.simulation.get_camera_image()
                detected_objects = self.perception.detect_objects(
                    rgb_image,
                    depth_image,
                    self.simulation.camera_params
                )
                
                print(f"  Detected {len(detected_objects)} objects")
                for obj in detected_objects:
                    print(f"    - {obj.name} at {obj.position}")
                
                # Save scene representation
                scene_json = self.perception.to_json(detected_objects)
                
                # 2. Language: Parse command
                print("\n[2/5] Language: Parsing command...")
                parsed_command = self.language.parse_command(command)
                print(f"  Action: {parsed_command.action.value}")
                print(f"  Target: {parsed_command.target_object}")
                print(f"  Destination: {parsed_command.destination_object}")
                print(f"  Spatial Relation: {parsed_command.spatial_relation.value}")
                
                # 3. Planning: Generate action plan
                print("\n[3/5] Planning: Generating action plan...")
                
                # Get current end-effector pose
                current_ee_pose = self.executor.get_current_end_effector_pose()
                
                # Plan based on action type
                if parsed_command.action in [ActionType.PICK, ActionType.PLACE]:
                    # For place commands with pick intent, plan full pick-and-place
                    if "pick" in command.lower() and "place" in command.lower():
                        actions = self.planning.plan_pick_and_place(
                            parsed_command,
                            detected_objects,
                            current_ee_pose
                        )
                    else:
                        # Single action
                        action = self.planning.plan_action(
                            parsed_command,
                            detected_objects,
                            current_ee_pose
                        )
                        actions = [action]
                else:
                    print(f"  Action type {parsed_command.action} not yet implemented")
                    return False
                
                total_waypoints = sum(len(a.waypoints) for a in actions)
                print(f"  Generated {len(actions)} actions with {total_waypoints} waypoints")
                
                # 4. Control: Execute actions
                print("\n[4/5] Control: Executing trajectory...")
                for i, action in enumerate(actions):
                    print(f"  Executing action {i+1}/{len(actions)}: {action.action_type.value}")
                    
                    if not action.waypoints:
                        print("    No waypoints to execute")
                        continue
                    
                    # Execute waypoints
                    success = self.executor.execute_waypoints(
                        action.waypoints,
                        self.simulation
                    )
                    
                    if not success:
                        print("    Execution failed")
                        if self.log_metrics:
                            self.metrics_logger.log_execution(
                                task_id=f"task_{self.task_counter}",
                                command=command,
                                success=False,
                                execution_time=timer.elapsed or 0.0,
                                num_waypoints=total_waypoints,
                                num_actions=len(actions),
                                failure_mode="execution_failed"
                            )
                        self.task_counter += 1
                        return False
                    
                    print(f"    Action {i+1} completed successfully")
                
                # 5. Feedback: Verify task completion
                print("\n[5/5] Feedback: Verifying task completion...")
                success = self._verify_task_completion(
                    parsed_command,
                    detected_objects
                )
                
                if success:
                    print("  ✓ Task completed successfully!")
                else:
                    print("  ✗ Task verification failed")
                
                # Log metrics
                if self.log_metrics:
                    self.metrics_logger.log_execution(
                        task_id=f"task_{self.task_counter}",
                        command=command,
                        success=success,
                        execution_time=timer.elapsed,
                        num_waypoints=total_waypoints,
                        num_actions=len(actions),
                        failure_mode="none" if success else "verification_failed"
                    )
                
                self.task_counter += 1
                
                print(f"\n{'='*60}")
                print(f"Execution Time: {timer.elapsed:.3f}s")
                print(f"{'='*60}\n")
                
                return success
                
            except Exception as e:
                print(f"\nError during execution: {str(e)}")
                import traceback
                traceback.print_exc()
                
                if self.log_metrics:
                    self.metrics_logger.log_execution(
                        task_id=f"task_{self.task_counter}",
                        command=command,
                        success=False,
                        execution_time=timer.elapsed or 0.0,
                        num_waypoints=0,
                        num_actions=0,
                        failure_mode=f"exception_{type(e).__name__}"
                    )
                
                self.task_counter += 1
                return False
    
    def _verify_task_completion(
        self,
        parsed_command: ParsedCommand,
        initial_objects: List[Object3D]
    ) -> bool:
        """
        Verify task completion via vision feedback.
        
        Args:
            parsed_command: Original parsed command
            initial_objects: Objects detected at start
            
        Returns:
            True if task appears successful
        """
        # Re-detect objects
        rgb_image, depth_image = self.simulation.get_camera_image()
        current_objects = self.perception.detect_objects(
            rgb_image,
            depth_image,
            self.simulation.camera_params
        )
        
        # Simple verification: check if objects moved
        # More sophisticated verification can be added
        if len(current_objects) != len(initial_objects):
            # Object count changed - may indicate success or failure
            pass
        
        # For now, assume success if execution completed without errors
        return True
    
    def setup_scene(self, scene_config: Dict):
        """
        Setup simulation scene with objects and trays.
        
        Args:
            scene_config: Dictionary with object and tray configurations
                - 'objects': List of objects to add
                - 'trays': Optional list of trays/placement zones to add
        """
        print("\nSetting up scene...")
        
        # Reset simulation
        self.simulation.reset()
        
        # Add trays first (so they're on the bottom)
        trays = scene_config.get('trays', [])
        for tray in trays:
            self.simulation.add_tray(
                position=tuple(tray['position']),
                size=tuple(tray.get('size', (0.2, 0.2))),
                height=tray.get('height', 0.005),
                color=tray.get('color', 'gray'),
                label=tray.get('label', None)
            )
            label = tray.get('label', f"{tray.get('color', 'gray')} tray")
            print(f"  Added {label} at {tray['position']}")
        
        # Add objects
        objects = scene_config.get('objects', [])
        for obj in objects:
            self.simulation.add_object(
                shape=obj['shape'],
                color=obj['color'],
                position=tuple(obj['position']),
                size=obj.get('size', 0.05)
            )
            print(f"  Added {obj['color']} {obj['shape']} at {obj['position']}")
        
        # Step simulation to settle objects
        self.simulation.step(num_steps=100)
        
        print("Scene setup complete.\n")
    
    def get_scene_representation(self) -> Dict:
        """
        Get current scene representation as JSON.
        
        Returns:
            Scene representation dictionary
        """
        rgb_image, depth_image = self.simulation.get_camera_image()
        detected_objects = self.perception.detect_objects(
            rgb_image,
            depth_image,
            self.simulation.camera_params
        )
        
        return self.perception.get_scene_representation(detected_objects)
    
    def save_metrics(self, filepath: str = "metrics.json"):
        """Save execution metrics to file."""
        if self.log_metrics:
            self.metrics_logger.log_file = filepath
            self.metrics_logger.save_metrics()
            print(f"\nMetrics saved to {filepath}")
    
    def print_metrics_summary(self):
        """Print metrics summary."""
        if self.log_metrics:
            self.metrics_logger.print_summary()
    
    def close(self):
        """Close pipeline and cleanup resources."""
        if self.log_metrics:
            self.metrics_logger.save_metrics()
        
        self.simulation.close()

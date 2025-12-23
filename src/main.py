#!/usr/bin/env python3
"""
Advanced Computer Vision System - Main Application

This application integrates state-of-the-art computer vision models from recent
CVPR, ICCV, and ECCV conferences to provide comprehensive real-time video analysis.

Features:
- Advanced object detection (YOLO v8/v9, DINO)
- Semantic/instance segmentation (SAM, Mask2Former)
- Monocular depth estimation (DPT, MiDaS)
- Human pose estimation (ViTPose, MediaPipe)
- Multi-object tracking (MOT)
- Scene understanding (CLIP, BLIP-2)
- Facial analysis (emotion, age, gender)
- Hand pose and gesture recognition
"""

import cv2
import numpy as np
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_system.camera import CameraInterface
from cv_system.models import (
    ObjectDetector,
    SegmentAnything,
    SemanticSegmentation,
    DepthEstimator,
    PoseEstimator,
    HandPoseEstimator,
    CLIPModel,
    BLIP2,
    SceneUnderstanding,
    FaceAnalyzer,
    MultiObjectTracker,
)
from cv_system.utils import (
    Visualizer,
    FPSCounter,
    PerformanceMonitor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedCVSystem:
    """
    Main computer vision system integrating multiple SOTA models
    """
    
    def __init__(self, config: dict):
        """
        Initialize CV system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.camera = None
        self.visualizer = Visualizer()
        self.fps_counter = FPSCounter()
        self.perf_monitor = PerformanceMonitor()
        
        # Initialize models based on config
        self.models = {}
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize selected models"""
        logger.info("Initializing models...")
        
        device = self.config.get('device', 'auto')
        
        # Object Detection
        if self.config.get('enable_detection', True):
            try:
                logger.info("Loading object detection model...")
                model_type = self.config.get('detection_model', 'yolov8n')  # Using nano for faster loading
                self.models['detector'] = ObjectDetector(model_type=model_type, device=device)
                self.models['tracker'] = MultiObjectTracker()
                logger.info("Object detection ready")
            except Exception as e:
                logger.error(f"Failed to load detector: {e}")
        
        # Depth Estimation
        if self.config.get('enable_depth', False):
            try:
                logger.info("Loading depth estimation model...")
                self.models['depth'] = DepthEstimator(device=device)
                logger.info("Depth estimation ready")
            except Exception as e:
                logger.warning(f"Failed to load depth estimator: {e}")
        
        # Pose Estimation
        if self.config.get('enable_pose', True):
            try:
                logger.info("Loading pose estimation model...")
                self.models['pose'] = PoseEstimator(device=device)
                logger.info("Pose estimation ready")
            except Exception as e:
                logger.warning(f"Failed to load pose estimator: {e}")
        
        # Hand Pose
        if self.config.get('enable_hands', False):
            try:
                logger.info("Loading hand pose estimation...")
                self.models['hands'] = HandPoseEstimator(device=device)
                logger.info("Hand pose estimation ready")
            except Exception as e:
                logger.warning(f"Failed to load hand pose: {e}")
        
        # Face Analysis
        if self.config.get('enable_face', True):
            try:
                logger.info("Loading face analysis...")
                self.models['face'] = FaceAnalyzer(device=device)
                logger.info("Face analysis ready")
            except Exception as e:
                logger.warning(f"Failed to load face analyzer: {e}")
        
        # Scene Understanding
        if self.config.get('enable_scene', False):
            try:
                logger.info("Loading scene understanding...")
                self.models['scene'] = SceneUnderstanding(device=device)
                logger.info("Scene understanding ready")
            except Exception as e:
                logger.warning(f"Failed to load scene understanding: {e}")
        
        logger.info(f"Initialized {len(self.models)} model modules")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process single frame through all enabled models
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        # Object Detection
        if 'detector' in self.models:
            self.perf_monitor.start_timer('detection')
            detections = self.models['detector'].detect(frame, conf_threshold=0.5)
            results['detections'] = detections
            self.perf_monitor.stop_timer('detection')
            
            # Tracking
            if 'tracker' in self.models and detections:
                self.perf_monitor.start_timer('tracking')
                tracks = self.models['tracker'].update(detections)
                results['tracks'] = tracks
                self.perf_monitor.stop_timer('tracking')
        
        # Depth Estimation
        if 'depth' in self.models and self.config.get('enable_depth', False):
            self.perf_monitor.start_timer('depth')
            depth_map = self.models['depth'].estimate_depth(frame)
            results['depth'] = depth_map
            self.perf_monitor.stop_timer('depth')
        
        # Pose Estimation
        if 'pose' in self.models:
            self.perf_monitor.start_timer('pose')
            poses = self.models['pose'].estimate_pose(frame)
            results['poses'] = poses
            self.perf_monitor.stop_timer('pose')
        
        # Hand Pose
        if 'hands' in self.models and self.config.get('enable_hands', False):
            self.perf_monitor.start_timer('hands')
            hands = self.models['hands'].detect_hands(frame)
            results['hands'] = hands
            self.perf_monitor.stop_timer('hands')
        
        # Face Analysis
        if 'face' in self.models:
            self.perf_monitor.start_timer('face')
            faces = self.models['face'].detect_faces(frame)
            results['faces'] = faces
            self.perf_monitor.stop_timer('face')
        
        return results
    
    def visualize_results(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        Visualize all results on frame
        
        Args:
            frame: Original frame
            results: Processing results
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Visualize detections
        if 'detections' in results and 'detector' in self.models:
            vis_frame = self.models['detector'].visualize(vis_frame, results['detections'])
        
        # Visualize tracks
        if 'tracks' in results and 'tracker' in self.models:
            vis_frame = self.models['tracker'].visualize_tracks(vis_frame, results['tracks'])
        
        # Visualize poses
        if 'poses' in results and 'pose' in self.models:
            vis_frame = self.models['pose'].visualize_poses(vis_frame, results['poses'])
        
        # Visualize faces
        if 'faces' in results and 'face' in self.models:
            vis_frame = self.models['face'].visualize_faces(vis_frame, results['faces'])
        
        # Add FPS counter
        fps = self.fps_counter.tick()
        vis_frame = self.visualizer.draw_fps(vis_frame, fps)
        
        # Add info panel
        info = {
            'Objects': len(results.get('detections', [])),
            'Tracks': len(results.get('tracks', {})),
            'Poses': len(results.get('poses', [])),
            'Faces': len(results.get('faces', []))
        }
        vis_frame = self.visualizer.draw_info_panel(vis_frame, info)
        
        return vis_frame
    
    def run(self):
        """Run main processing loop"""
        logger.info("Starting CV system...")
        
        # Initialize camera
        camera_id = self.config.get('camera_id', 0)
        resolution = self.config.get('resolution', (1280, 720))
        fps = self.config.get('fps', 30)
        
        self.camera = CameraInterface(camera_id=camera_id, resolution=resolution, fps=fps)
        
        if not self.camera.open():
            logger.error("Failed to open camera")
            return
        
        logger.info("Camera ready. Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Visualize
                vis_frame = self.visualize_results(frame, results)
                
                # Display
                cv2.imshow('Advanced CV System', vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save frame
                    filename = f"output/frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    logger.info(f"Saved {filename}")
                elif key == ord('p'):
                    # Print performance stats
                    self.perf_monitor.print_stats()
                
                frame_count += 1
                
                # Periodic logging
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            logger.info("Cleaning up...")
            self.camera.release()
            cv2.destroyAllWindows()
            self.perf_monitor.print_stats()
            logger.info("Done")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced Computer Vision System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py
  
  # Run with specific camera
  python main.py --camera 1
  
  # Enable all features
  python main.py --enable-depth --enable-scene --enable-hands
  
  # Use specific detection model
  python main.py --detection-model yolov8x
        """
    )
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                       help='Camera resolution (default: 1280x720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run on (default: auto)')
    parser.add_argument('--detection-model', type=str, default='yolov8n',
                       help='Object detection model (default: yolov8n)')
    parser.add_argument('--enable-depth', action='store_true',
                       help='Enable depth estimation')
    parser.add_argument('--enable-scene', action='store_true',
                       help='Enable scene understanding')
    parser.add_argument('--enable-hands', action='store_true',
                       help='Enable hand pose estimation')
    parser.add_argument('--no-detection', action='store_true',
                       help='Disable object detection')
    parser.add_argument('--no-pose', action='store_true',
                       help='Disable pose estimation')
    parser.add_argument('--no-face', action='store_true',
                       help='Disable face analysis')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Build config
    config = {
        'camera_id': args.camera,
        'resolution': (width, height),
        'fps': args.fps,
        'device': args.device,
        'detection_model': args.detection_model,
        'enable_detection': not args.no_detection,
        'enable_depth': args.enable_depth,
        'enable_pose': not args.no_pose,
        'enable_scene': args.enable_scene,
        'enable_hands': args.enable_hands,
        'enable_face': not args.no_face,
    }
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    # Run system
    system = AdvancedCVSystem(config)
    system.run()


if __name__ == '__main__':
    main()

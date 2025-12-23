#!/usr/bin/env python3
"""
Comprehensive Image Analysis Script

This script takes any image as input and provides ALL possible information
with high confidence, including:
- Object detection and classification
- Scene understanding and categorization
- Depth estimation
- Face analysis (emotion, age, gender, landmarks)
- Pose estimation for all people
- Hand detection and poses
- Image captioning
- Semantic segmentation
- Text descriptions
- Color analysis
- Image quality metrics
"""

import sys
import os
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from typing import Dict, List, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_system.models import (
    ObjectDetector,
    SemanticSegmentation,
    DepthEstimator,
    PoseEstimator,
    HandPoseEstimator,
    CLIPModel,
    BLIP2,
    SceneUnderstanding,
    FaceAnalyzer,
)
from cv_system.utils import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveImageAnalyzer:
    """
    Comprehensive image analysis system that extracts all possible information
    from a single image with high confidence.
    """
    
    def __init__(self, device: str = "auto", high_confidence: bool = True):
        """
        Initialize comprehensive analyzer
        
        Args:
            device: Device to run on (cuda, cpu, or auto)
            high_confidence: Use high confidence thresholds
        """
        self.device = device
        self.high_confidence = high_confidence
        self.min_confidence = 0.7 if high_confidence else 0.5
        
        logger.info("Initializing Comprehensive Image Analyzer...")
        logger.info(f"Device: {device}, High Confidence Mode: {high_confidence}")
        
        # Initialize all models
        self.models = {}
        self._initialize_models()
        
        self.visualizer = Visualizer()
        
    def _initialize_models(self):
        """Initialize all available models"""
        
        # Object Detection (high accuracy model)
        try:
            logger.info("Loading object detection model (YOLOv8x for high accuracy)...")
            self.models['detector'] = ObjectDetector(
                model_type='yolov8x',  # Highest accuracy
                device=self.device
            )
            logger.info("✓ Object detection ready")
        except Exception as e:
            logger.warning(f"Object detection unavailable: {e}")
        
        # Scene Understanding
        try:
            logger.info("Loading scene understanding (CLIP)...")
            self.models['scene'] = SceneUnderstanding(device=self.device)
            logger.info("✓ Scene understanding ready")
        except Exception as e:
            logger.warning(f"Scene understanding unavailable: {e}")
        
        # Image Captioning
        try:
            logger.info("Loading image captioning (BLIP-2)...")
            self.models['caption'] = BLIP2(device=self.device)
            logger.info("✓ Image captioning ready")
        except Exception as e:
            logger.warning(f"Image captioning unavailable: {e}")
        
        # Depth Estimation
        try:
            logger.info("Loading depth estimation (DPT-Large)...")
            self.models['depth'] = DepthEstimator(
                model_type='dpt-large',
                device=self.device
            )
            logger.info("✓ Depth estimation ready")
        except Exception as e:
            logger.warning(f"Depth estimation unavailable: {e}")
        
        # Pose Estimation
        try:
            logger.info("Loading pose estimation...")
            self.models['pose'] = PoseEstimator(device=self.device)
            logger.info("✓ Pose estimation ready")
        except Exception as e:
            logger.warning(f"Pose estimation unavailable: {e}")
        
        # Hand Detection
        try:
            logger.info("Loading hand detection...")
            self.models['hands'] = HandPoseEstimator(device=self.device)
            logger.info("✓ Hand detection ready")
        except Exception as e:
            logger.warning(f"Hand detection unavailable: {e}")
        
        # Face Analysis
        try:
            logger.info("Loading face analysis...")
            self.models['face'] = FaceAnalyzer(device=self.device)
            logger.info("✓ Face analysis ready")
        except Exception as e:
            logger.warning(f"Face analysis unavailable: {e}")
        
        # Semantic Segmentation
        try:
            logger.info("Loading semantic segmentation...")
            self.models['segmentation'] = SemanticSegmentation(device=self.device)
            logger.info("✓ Semantic segmentation ready")
        except Exception as e:
            logger.warning(f"Segmentation unavailable: {e}")
        
        logger.info(f"Initialized {len(self.models)} model modules")
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Analyzing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = image.shape[:2]
        logger.info(f"Image size: {w}x{h}")
        
        results = {
            'image_path': image_path,
            'image_size': {'width': w, 'height': h},
            'models_used': list(self.models.keys()),
        }
        
        # 1. Object Detection
        if 'detector' in self.models:
            logger.info("Running object detection...")
            detections = self.models['detector'].detect(
                image, 
                conf_threshold=self.min_confidence
            )
            results['objects'] = {
                'count': len(detections),
                'detections': detections,
                'categories': list(set([d['class'] for d in detections])),
                'confidence_threshold': self.min_confidence
            }
            logger.info(f"Found {len(detections)} objects")
        
        # 2. Scene Understanding
        if 'scene' in self.models:
            logger.info("Analyzing scene...")
            scene_analysis = self.models['scene'].analyze_scene(image)
            results['scene'] = scene_analysis
            logger.info(f"Scene type: {scene_analysis.get('scene_type', 'N/A')}")
        
        # 3. Image Captioning
        if 'caption' in self.models:
            logger.info("Generating image caption...")
            caption = self.models['caption'].generate_caption(image)
            results['caption'] = caption
            logger.info(f"Caption: {caption}")
        
        # 4. Depth Estimation
        if 'depth' in self.models:
            logger.info("Estimating depth...")
            depth_map = self.models['depth'].estimate_depth(image)
            if depth_map is not None:
                results['depth'] = {
                    'available': True,
                    'min_depth': float(depth_map.min()),
                    'max_depth': float(depth_map.max()),
                    'mean_depth': float(depth_map.mean()),
                    'depth_map_shape': depth_map.shape
                }
                logger.info("Depth estimation complete")
        
        # 5. Human Pose Analysis
        if 'pose' in self.models:
            logger.info("Detecting human poses...")
            poses = self.models['pose'].estimate_pose(image)
            results['poses'] = {
                'count': len(poses),
                'people': poses
            }
            logger.info(f"Found {len(poses)} people")
        
        # 6. Hand Detection
        if 'hands' in self.models:
            logger.info("Detecting hands...")
            hands = self.models['hands'].detect_hands(image)
            results['hands'] = {
                'count': len(hands),
                'detections': hands
            }
            logger.info(f"Found {len(hands)} hands")
        
        # 7. Face Analysis
        if 'face' in self.models:
            logger.info("Analyzing faces...")
            faces = self.models['face'].detect_faces(image, min_confidence=self.min_confidence)
            
            # Analyze each face in detail
            face_analyses = []
            for face in faces:
                analysis = self.models['face'].analyze_face(image, face['bbox'])
                face_analyses.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    **analysis
                })
            
            results['faces'] = {
                'count': len(faces),
                'analyses': face_analyses
            }
            logger.info(f"Analyzed {len(faces)} faces")
        
        # 8. Image Quality & Statistics
        results['image_statistics'] = self._analyze_image_statistics(image)
        
        # 9. Color Analysis
        results['colors'] = self._analyze_colors(image)
        
        # 10. Additional CLIP-based classifications
        if 'scene' in self.models and hasattr(self.models['scene'], 'clip_model'):
            results['detailed_classification'] = self._detailed_classification(image)
        
        logger.info("Analysis complete!")
        return results
    
    def _analyze_image_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze basic image statistics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        stats = {
            'brightness': {
                'mean': float(gray.mean()),
                'std': float(gray.std()),
                'min': int(gray.min()),
                'max': int(gray.max())
            },
            'contrast': float(gray.std()),
            'sharpness': self._calculate_sharpness(gray),
        }
        
        return stats
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze dominant colors in image"""
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape to list of pixels
        pixels = rgb.reshape(-1, 3)
        
        # Calculate color statistics
        color_stats = {
            'mean_rgb': [float(x) for x in pixels.mean(axis=0)],
            'std_rgb': [float(x) for x in pixels.std(axis=0)],
            'dominant_hue': self._get_dominant_hue(image)
        }
        
        return color_stats
    
    def _get_dominant_hue(self, image: np.ndarray) -> str:
        """Get dominant hue category"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        
        mean_hue = hue.mean()
        
        # Categorize hue
        if mean_hue < 15 or mean_hue > 165:
            return "red"
        elif mean_hue < 45:
            return "orange/yellow"
        elif mean_hue < 75:
            return "green"
        elif mean_hue < 105:
            return "cyan"
        elif mean_hue < 135:
            return "blue"
        else:
            return "magenta/purple"
    
    def _detailed_classification(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform detailed classification with multiple categories"""
        
        # Object categories
        object_categories = [
            "person", "animal", "vehicle", "furniture", "electronics",
            "food", "building", "nature", "indoor object", "outdoor object"
        ]
        
        # Style categories
        style_categories = [
            "photograph", "painting", "drawing", "digital art", "sketch",
            "3D render", "cartoon", "abstract art"
        ]
        
        # Quality categories
        quality_categories = [
            "high quality", "professional", "amateur", "low quality",
            "well lit", "poorly lit", "blurry", "sharp"
        ]
        
        scene = self.models['scene']
        
        classifications = {
            'object_types': scene.clip_model.classify_image(image, object_categories),
            'style': scene.clip_model.classify_image(image, style_categories),
            'quality': scene.clip_model.classify_image(image, quality_categories)
        }
        
        return classifications
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_path}")
    
    def visualize_results(self, image_path: str, results: Dict[str, Any], 
                         output_dir: str = "output"):
        """
        Create comprehensive visualization of all results
        
        Args:
            image_path: Path to original image
            results: Analysis results
            output_dir: Output directory for visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image = cv2.imread(image_path)
        base_name = Path(image_path).stem
        
        visualizations = []
        labels = []
        
        # 1. Original image with detections
        if 'objects' in results and 'detector' in self.models:
            vis_detection = self.models['detector'].visualize(
                image.copy(), 
                results['objects']['detections']
            )
            visualizations.append(vis_detection)
            labels.append(f"Objects: {results['objects']['count']}")
        
        # 2. Pose visualization
        if 'poses' in results and 'pose' in self.models:
            vis_pose = self.models['pose'].visualize_poses(
                image.copy(),
                results['poses']['people']
            )
            visualizations.append(vis_pose)
            labels.append(f"Poses: {results['poses']['count']}")
        
        # 3. Face visualization
        if 'faces' in results and 'face' in self.models:
            vis_face = self.models['face'].visualize_faces(
                image.copy(),
                [{'bbox': f['bbox'], 'confidence': f['confidence']} 
                 for f in results['faces']['analyses']],
                results['faces']['analyses']
            )
            visualizations.append(vis_face)
            labels.append(f"Faces: {results['faces']['count']}")
        
        # 4. Depth visualization
        if 'depth' in results and results['depth']['available'] and 'depth' in self.models:
            depth_map = self.models['depth'].estimate_depth(image)
            if depth_map is not None:
                vis_depth = self.models['depth'].visualize_depth(depth_map)
                visualizations.append(vis_depth)
                labels.append("Depth Map")
        
        # Save individual visualizations
        for i, (vis, label) in enumerate(zip(visualizations, labels)):
            output_path = os.path.join(output_dir, f"{base_name}_vis_{i}_{label.replace(' ', '_').replace(':', '')}.jpg")
            cv2.imwrite(output_path, vis)
            logger.info(f"Saved: {output_path}")
        
        # Create grid visualization
        if len(visualizations) > 0:
            grid = self.visualizer.create_grid(visualizations, labels)
            grid_path = os.path.join(output_dir, f"{base_name}_comprehensive_analysis.jpg")
            cv2.imwrite(grid_path, grid)
            logger.info(f"Saved comprehensive grid: {grid_path}")
        
        # Save annotated text summary
        self._save_text_summary(results, os.path.join(output_dir, f"{base_name}_analysis.txt"))
    
    def _save_text_summary(self, results: Dict[str, Any], output_path: str):
        """Save human-readable text summary"""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE IMAGE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Image: {results['image_path']}\n")
            f.write(f"Size: {results['image_size']['width']}x{results['image_size']['height']}\n\n")
            
            # Caption
            if 'caption' in results:
                f.write(f"CAPTION: {results['caption']}\n\n")
            
            # Scene
            if 'scene' in results:
                f.write("SCENE ANALYSIS:\n")
                for key, (label, conf) in results['scene'].items():
                    f.write(f"  {key}: {label} ({conf:.2%} confidence)\n")
                f.write("\n")
            
            # Objects
            if 'objects' in results:
                f.write(f"OBJECTS DETECTED: {results['objects']['count']}\n")
                f.write(f"Categories: {', '.join(results['objects']['categories'])}\n")
                for obj in results['objects']['detections']:
                    f.write(f"  - {obj['class']}: {obj['confidence']:.2%} confidence\n")
                f.write("\n")
            
            # People
            if 'poses' in results:
                f.write(f"PEOPLE DETECTED: {results['poses']['count']}\n\n")
            
            # Faces
            if 'faces' in results:
                f.write(f"FACES ANALYZED: {results['faces']['count']}\n")
                for i, face in enumerate(results['faces']['analyses'], 1):
                    f.write(f"  Face {i}:\n")
                    if face.get('emotion'):
                        f.write(f"    Emotion: {face['emotion']}\n")
                    if face.get('age'):
                        f.write(f"    Age: ~{face['age']} years\n")
                    if face.get('gender'):
                        f.write(f"    Gender: {face['gender']}\n")
                f.write("\n")
            
            # Image Quality
            if 'image_statistics' in results:
                stats = results['image_statistics']
                f.write("IMAGE QUALITY:\n")
                f.write(f"  Brightness: {stats['brightness']['mean']:.1f}/255\n")
                f.write(f"  Contrast: {stats['contrast']:.1f}\n")
                f.write(f"  Sharpness: {stats['sharpness']:.1f}\n\n")
            
            # Colors
            if 'colors' in results:
                f.write("COLOR ANALYSIS:\n")
                f.write(f"  Dominant hue: {results['colors']['dominant_hue']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Analysis complete. All available information extracted with high confidence.\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Text summary saved: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Image Analysis - Extract ALL information from any image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python src/analyze_image.py input.jpg
  
  # Analyze with CPU
  python src/analyze_image.py input.jpg --device cpu
  
  # Save results to specific directory
  python src/analyze_image.py input.jpg --output results/
  
  # Standard confidence mode (faster)
  python src/analyze_image.py input.jpg --standard-confidence
        """
    )
    
    parser.add_argument('image', type=str,
                       help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run on (default: auto)')
    parser.add_argument('--standard-confidence', action='store_true',
                       help='Use standard confidence thresholds (faster)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--json-only', action='store_true',
                       help='Only save JSON results, skip visualizations')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize analyzer
    high_confidence = not args.standard_confidence
    analyzer = ComprehensiveImageAnalyzer(
        device=args.device,
        high_confidence=high_confidence
    )
    
    # Analyze image
    try:
        results = analyzer.analyze_image(args.image)
        
        # Save JSON results
        base_name = Path(args.image).stem
        json_path = os.path.join(args.output, f"{base_name}_analysis.json")
        analyzer.save_results(results, json_path)
        
        # Generate visualizations
        if not args.json_only and not args.no_visualize:
            analyzer.visualize_results(args.image, results, args.output)
        
        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Image: {args.image}")
        print(f"Models used: {len(results['models_used'])}")
        
        if 'caption' in results:
            print(f"\nCaption: {results['caption']}")
        
        if 'objects' in results:
            print(f"\nObjects: {results['objects']['count']} detected")
            print(f"Categories: {', '.join(results['objects']['categories'][:5])}")
        
        if 'faces' in results:
            print(f"\nFaces: {results['faces']['count']} analyzed")
        
        if 'poses' in results:
            print(f"People: {results['poses']['count']} detected")
        
        print(f"\nResults saved to: {args.output}/")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""
Perception Module

Detects objects in simulation using computer vision techniques.
Outputs structured scene representation.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class Object3D:
    """Represents a detected 3D object in the scene."""
    id: int
    name: str
    color: str
    shape: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    size: Tuple[float, float, float]
    confidence: float = 1.0


class PerceptionModule:
    """
    Computer vision-based perception module for object detection.
    
    Uses color-based segmentation and contour analysis to detect objects
    in simulation. Designed to be modular and extensible.
    """
    
    def __init__(self, color_ranges: Optional[Dict[str, Dict]] = None):
        """
        Initialize perception module.
        
        Args:
            color_ranges: Dictionary mapping color names to HSV ranges
        """
        # Define color ranges in HSV space
        self.color_ranges = color_ranges or {
            'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'red2': {'lower': np.array([170, 100, 100]), 'upper': np.array([180, 255, 255])},
            'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
            'green': {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])},
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([40, 255, 255])},
        }
        
        self.shape_detector = ShapeDetector()
    
    def detect_objects(
        self, 
        rgb_image: np.ndarray, 
        depth_image: Optional[np.ndarray] = None,
        camera_params: Optional[Dict] = None
    ) -> List[Object3D]:
        """
        Detect objects in the scene from RGB and depth images.
        
        Args:
            rgb_image: RGB image from camera (H, W, 3)
            depth_image: Depth image from camera (H, W)
            camera_params: Camera intrinsics for 3D reconstruction
            
        Returns:
            List of detected Object3D instances
        """
        detected_objects = []
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        object_id = 0
        for color_name, color_range in self.color_ranges.items():
            if color_name == 'red2':  # Handle red wrap-around
                continue
                
            # Create color mask
            mask = self._create_color_mask(hsv_image, color_name)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Filter small noise
                    continue
                
                # Detect shape
                shape = self.shape_detector.detect(contour)
                
                # Get bounding box and center
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2
                
                # Estimate 3D position
                position = self._estimate_3d_position(
                    center_x, center_y, depth_image, camera_params
                )
                
                # Estimate size
                size = self._estimate_size(w, h, depth_image, camera_params)
                
                obj = Object3D(
                    id=object_id,
                    name=f"{color_name}_{shape}",
                    color=color_name,
                    shape=shape,
                    position=position,
                    orientation=(0, 0, 0, 1),  # Default orientation
                    size=size,
                    confidence=0.95
                )
                detected_objects.append(obj)
                object_id += 1
        
        return detected_objects
    
    def _create_color_mask(self, hsv_image: np.ndarray, color_name: str) -> np.ndarray:
        """Create binary mask for specific color."""
        color_range = self.color_ranges[color_name]
        mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
        
        # Handle red wrap-around
        if color_name == 'red' and 'red2' in self.color_ranges:
            mask2 = cv2.inRange(
                hsv_image, 
                self.color_ranges['red2']['lower'], 
                self.color_ranges['red2']['upper']
            )
            mask = cv2.bitwise_or(mask, mask2)
        
        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _estimate_3d_position(
        self, 
        pixel_x: int, 
        pixel_y: int, 
        depth_image: Optional[np.ndarray],
        camera_params: Optional[Dict]
    ) -> Tuple[float, float, float]:
        """Estimate 3D position from pixel coordinates and depth."""
        if depth_image is None or camera_params is None:
            # Default fallback - return normalized image coordinates
            return (pixel_x / 640.0 - 0.5, pixel_y / 480.0 - 0.5, 0.0)
        
        # Get depth value
        depth = depth_image[pixel_y, pixel_x]
        
        # Unproject to 3D using camera intrinsics
        fx = camera_params.get('fx', 500)
        fy = camera_params.get('fy', 500)
        cx = camera_params.get('cx', 320)
        cy = camera_params.get('cy', 240)
        
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth
        
        return (x, y, z)
    
    def _estimate_size(
        self,
        width: int,
        height: int,
        depth_image: Optional[np.ndarray],
        camera_params: Optional[Dict]
    ) -> Tuple[float, float, float]:
        """Estimate object size from bounding box."""
        if depth_image is None or camera_params is None:
            # Default cube size
            return (0.05, 0.05, 0.05)
        
        # Estimate based on pixel size and depth
        avg_depth = 1.0  # Default depth
        scale_factor = avg_depth / 500.0  # Approximate scaling
        
        size_x = width * scale_factor
        size_y = height * scale_factor
        size_z = min(size_x, size_y)  # Assume cube-like objects
        
        return (size_x, size_y, size_z)
    
    def to_json(self, objects: List[Object3D]) -> str:
        """Convert detected objects to JSON representation."""
        objects_dict = [asdict(obj) for obj in objects]
        return json.dumps(objects_dict, indent=2)
    
    def get_scene_representation(self, objects: List[Object3D]) -> Dict:
        """Get structured scene representation."""
        return {
            'timestamp': None,  # Can be added if needed
            'num_objects': len(objects),
            'objects': [asdict(obj) for obj in objects]
        }


class ShapeDetector:
    """Detects geometric shapes from contours."""
    
    def detect(self, contour: np.ndarray) -> str:
        """
        Detect shape from contour.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Shape name (cube, sphere, cylinder)
        """
        # Approximate contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Get aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Determine shape
        num_vertices = len(approx)
        
        if num_vertices == 4 and 0.9 <= aspect_ratio <= 1.1:
            return "cube"
        elif num_vertices > 8:  # Many vertices indicate circle/sphere
            return "sphere"
        elif num_vertices >= 4 and abs(aspect_ratio - 1.0) > 0.2:
            return "cylinder"
        else:
            return "cube"  # Default

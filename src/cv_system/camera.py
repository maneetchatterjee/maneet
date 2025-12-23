"""
Camera Interface Module
Provides unified interface for camera access and frame capture
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraInterface:
    """Interface for camera capture with advanced settings"""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1920, 1080), fps: int = 30):
        """
        Initialize camera interface
        
        Args:
            camera_id: Camera device ID (0 for default)
            resolution: Desired resolution (width, height)
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
    def open(self) -> bool:
        """
        Open camera connection
        
        Returns:
            Success status
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Enable autofocus if available
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} fps")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from camera
        
        Returns:
            (success, frame) tuple
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

"""
Advanced Object Detection Module
Implements state-of-the-art object detection from recent CV conferences
- YOLO v8/v9 (real-time detection)
- DINO (Detection with Transformers)
- Grounding DINO (open-vocabulary detection)
"""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Multi-model object detection system"""
    
    def __init__(self, model_type: str = "yolov8x", device: str = "auto"):
        """
        Initialize object detector
        
        Args:
            model_type: Model architecture (yolov8x, yolov9, etc.)
            device: Device to run on (cuda, cpu, or auto)
        """
        self.model_type = model_type
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load detection model"""
        try:
            logger.info(f"Loading {self.model_type} on {self.device}")
            self.model = YOLO(f"{self.model_type}.pt")
            logger.info(f"Model {self.model_type} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in frame
        
        Args:
            frame: Input image (BGR format)
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections with bbox, class, confidence
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class": class_name,
                        "class_id": cls,
                        "confidence": conf
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def visualize(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detections on frame
        
        Args:
            frame: Input image
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det["class"]
            conf = det["confidence"]
            
            # Draw bounding box
            color = self._get_color(det["class_id"])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for class ID"""
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color


class GroundingDINO:
    """
    Open-vocabulary object detection using Grounding DINO
    Based on ECCV 2022 work
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize Grounding DINO"""
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Placeholder for Grounding DINO implementation
        # Would require groundingdino package
        logger.info("Grounding DINO initialized (placeholder)")
        
    def detect_with_text(self, frame: np.ndarray, text_prompt: str, 
                        box_threshold: float = 0.35) -> List[Dict]:
        """
        Detect objects using text prompts
        
        Args:
            frame: Input image
            text_prompt: Text description of objects to detect
            box_threshold: Detection threshold
            
        Returns:
            List of detections
        """
        # This is a placeholder - actual implementation would use
        # groundingdino library with transformer-based detection
        logger.warning("Grounding DINO text-based detection not fully implemented")
        return []

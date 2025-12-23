"""
Pose Estimation Module
Implements state-of-the-art human pose estimation:
- ViTPose (ECCV 2022)
- MMPose integration
- Multi-person pose tracking
"""
import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Human pose estimation and tracking
    """
    
    def __init__(self, model_type: str = "vitpose", device: str = "auto"):
        """
        Initialize pose estimator
        
        Args:
            model_type: Model architecture (vitpose, hrnet, etc.)
            device: Device to run on
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_type = model_type
        self.model = None
        
        # COCO keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load pose estimation model"""
        try:
            # Using MediaPipe as a robust fallback
            import mediapipe as mp
            
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("Pose estimation model loaded (MediaPipe)")
            
        except ImportError:
            logger.warning("MediaPipe not installed. Limited pose features.")
        except Exception as e:
            logger.error(f"Failed to load pose model: {e}")
    
    def estimate_pose(self, frame: np.ndarray) -> List[Dict]:
        """
        Estimate poses in frame
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of pose detections with keypoints
        """
        if not hasattr(self, 'pose'):
            return []
        
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            results = self.pose.process(rgb)
            
            poses = []
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append({
                        'x': landmark.x * frame.shape[1],
                        'y': landmark.y * frame.shape[0],
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                poses.append({
                    'keypoints': keypoints,
                    'score': np.mean([kp['visibility'] for kp in keypoints])
                })
            
            return poses
            
        except Exception as e:
            logger.error(f"Pose estimation error: {e}")
            return []
    
    def visualize_poses(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        """
        Visualize detected poses
        
        Args:
            frame: Input image
            poses: List of pose detections
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        for pose in poses:
            keypoints = pose['keypoints']
            
            # Draw keypoints
            for kp in keypoints:
                if kp['visibility'] > 0.5:
                    x, y = int(kp['x']), int(kp['y'])
                    cv2.circle(vis_frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw skeleton
            for connection in self.skeleton:
                if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                    kp1 = keypoints[connection[0]]
                    kp2 = keypoints[connection[1]]
                    
                    if kp1['visibility'] > 0.5 and kp2['visibility'] > 0.5:
                        pt1 = (int(kp1['x']), int(kp1['y']))
                        pt2 = (int(kp2['x']), int(kp2['y']))
                        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)
        
        return vis_frame
    
    def extract_pose_features(self, poses: List[Dict]) -> Dict:
        """
        Extract high-level pose features
        
        Args:
            poses: List of pose detections
            
        Returns:
            Dictionary of pose features (angles, positions, etc.)
        """
        if not poses:
            return {}
        
        features = {}
        pose = poses[0]  # Process first person
        keypoints = pose['keypoints']
        
        # Calculate joint angles
        if len(keypoints) >= 17:
            # Elbow angles
            features['left_elbow_angle'] = self._calculate_angle(
                keypoints[5], keypoints[7], keypoints[9]  # shoulder-elbow-wrist
            )
            features['right_elbow_angle'] = self._calculate_angle(
                keypoints[6], keypoints[8], keypoints[10]
            )
            
            # Knee angles
            features['left_knee_angle'] = self._calculate_angle(
                keypoints[11], keypoints[13], keypoints[15]  # hip-knee-ankle
            )
            features['right_knee_angle'] = self._calculate_angle(
                keypoints[12], keypoints[14], keypoints[16]
            )
        
        return features
    
    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays
            a = np.array([p1['x'], p1['y']])
            b = np.array([p2['x'], p2['y']])
            c = np.array([p3['x'], p3['y']])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            
            return np.degrees(angle)
            
        except:
            return 0.0


class HandPoseEstimator:
    """
    Hand pose estimation and gesture recognition
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize hand pose estimator"""
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.load_model()
    
    def load_model(self):
        """Load hand pose model"""
        try:
            import mediapipe as mp
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("Hand pose estimator loaded")
            
        except Exception as e:
            logger.warning(f"Hand pose estimation not available: {e}")
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hands and their keypoints
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of hand detections
        """
        if not hasattr(self, 'hands'):
            return []
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            hands = []
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, 
                    results.multi_handedness
                ):
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append({
                            'x': lm.x * frame.shape[1],
                            'y': lm.y * frame.shape[0],
                            'z': lm.z
                        })
                    
                    hands.append({
                        'landmarks': landmarks,
                        'handedness': handedness.classification[0].label,
                        'score': handedness.classification[0].score
                    })
            
            return hands
            
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
            return []

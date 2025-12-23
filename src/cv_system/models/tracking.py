"""
Object Tracking Module
Advanced video object tracking:
- Multiple object tracking (MOT)
- Single object tracking (SOT)
- Re-identification
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MultiObjectTracker:
    """
    Multi-object tracking system
    Combines detection with tracking for robust MOT
    """
    
    def __init__(self, tracker_type: str = "sort"):
        """
        Initialize multi-object tracker
        
        Args:
            tracker_type: Tracking algorithm (sort, deepsort, bytetrack)
        """
        self.tracker_type = tracker_type
        self.tracks = {}
        self.next_id = 0
        self.max_age = 30  # Maximum frames to keep track without detection
        self.min_hits = 3  # Minimum detections before confirming track
        
        # Track history for visualization
        self.track_history = defaultdict(list)
        self.max_history_length = 30
        
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of object detections with bbox and class
            
        Returns:
            Dictionary of track_id -> track_info
        """
        # Simple tracking based on IoU matching
        # In production, would use more sophisticated methods
        
        active_tracks = {}
        
        if not detections:
            # Age out tracks
            for track_id, track in list(self.tracks.items()):
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
                    if track_id in self.track_history:
                        del self.track_history[track_id]
            return active_tracks
        
        # Match detections to existing tracks
        matched_tracks = set()
        
        for detection in detections:
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold
            
            # Find best matching track
            for track_id, track in self.tracks.items():
                iou = self._calculate_iou(detection['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                self.tracks[best_match_id]['bbox'] = detection['bbox']
                self.tracks[best_match_id]['class'] = detection.get('class', 'unknown')
                self.tracks[best_match_id]['confidence'] = detection.get('confidence', 1.0)
                self.tracks[best_match_id]['age'] = 0
                self.tracks[best_match_id]['hits'] += 1
                matched_tracks.add(best_match_id)
                
                # Update history
                center = self._get_center(detection['bbox'])
                self.track_history[best_match_id].append(center)
                if len(self.track_history[best_match_id]) > self.max_history_length:
                    self.track_history[best_match_id].pop(0)
                
                if self.tracks[best_match_id]['hits'] >= self.min_hits:
                    active_tracks[best_match_id] = self.tracks[best_match_id]
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'id': track_id,
                    'bbox': detection['bbox'],
                    'class': detection.get('class', 'unknown'),
                    'confidence': detection.get('confidence', 1.0),
                    'age': 0,
                    'hits': 1
                }
                
                center = self._get_center(detection['bbox'])
                self.track_history[track_id] = [center]
        
        # Age unmatched tracks
        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
                    if track_id in self.track_history:
                        del self.track_history[track_id]
        
        return active_tracks
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _get_center(self, bbox: List[float]) -> Tuple[int, int]:
        """Get bounding box center"""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def visualize_tracks(self, frame: np.ndarray, tracks: Dict[int, Dict]) -> np.ndarray:
        """
        Visualize tracks on frame
        
        Args:
            frame: Input frame
            tracks: Active tracks
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = map(int, track['bbox'])
            
            # Draw bounding box
            color = self._get_track_color(track_id)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and class
            label = f"ID:{track_id} {track['class']}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory
            if track_id in self.track_history:
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)
        
        return vis_frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID"""
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color


class SingleObjectTracker:
    """
    Single object tracking using OpenCV trackers
    """
    
    def __init__(self, tracker_type: str = "csrt"):
        """
        Initialize single object tracker
        
        Args:
            tracker_type: Tracker type (csrt, kcf, mosse, etc.)
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.initialized = False
    
    def initialize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        Initialize tracker with first frame and bbox
        
        Args:
            frame: Initial frame
            bbox: Bounding box (x, y, w, h)
        """
        # Create tracker
        if self.tracker_type == "csrt":
            self.tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == "kcf":
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == "mosse":
            self.tracker = cv2.legacy.TrackerMOSSE_create()
        else:
            logger.warning(f"Unknown tracker type: {self.tracker_type}, using CSRT")
            self.tracker = cv2.TrackerCSRT_create()
        
        # Initialize
        success = self.tracker.init(frame, bbox)
        self.initialized = success
        return success
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Update tracker with new frame
        
        Args:
            frame: New frame
            
        Returns:
            (success, bbox) tuple
        """
        if not self.initialized or self.tracker is None:
            return False, None
        
        success, bbox = self.tracker.update(frame)
        
        if success:
            return True, tuple(map(int, bbox))
        else:
            return False, None


class OpticalFlowTracker:
    """
    Optical flow-based tracking for motion analysis
    """
    
    def __init__(self):
        """Initialize optical flow tracker"""
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.prev_gray = None
        self.prev_points = None
    
    def initialize(self, frame: np.ndarray):
        """
        Initialize with first frame
        
        Args:
            frame: Initial frame (BGR)
        """
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_points = cv2.goodFeaturesToTrack(
            self.prev_gray,
            mask=None,
            **self.feature_params
        )
    
    def track_features(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Track features using optical flow
        
        Args:
            frame: New frame (BGR)
            
        Returns:
            (prev_points, curr_points) or (None, None) if tracking fails
        """
        if self.prev_gray is None or self.prev_points is None:
            self.initialize(frame)
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            **self.lk_params
        )
        
        if curr_points is not None:
            # Select good points
            good_prev = self.prev_points[status == 1]
            good_curr = curr_points[status == 1]
            
            # Update for next iteration
            self.prev_gray = gray.copy()
            self.prev_points = good_curr.reshape(-1, 1, 2)
            
            return good_prev, good_curr
        else:
            return None, None
    
    def visualize_flow(self, frame: np.ndarray, 
                       prev_points: np.ndarray, 
                       curr_points: np.ndarray) -> np.ndarray:
        """
        Visualize optical flow
        
        Args:
            frame: Current frame
            prev_points: Previous feature points
            curr_points: Current feature points
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        for prev, curr in zip(prev_points, curr_points):
            x1, y1 = prev.ravel()
            x2, y2 = curr.ravel()
            
            # Draw line showing motion
            cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(vis_frame, (int(x2), int(y2)), 5, (0, 0, 255), -1)
        
        return vis_frame

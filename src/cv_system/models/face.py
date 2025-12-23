"""
Facial Analysis Module
Comprehensive facial analysis including:
- Face detection and recognition
- Emotion recognition
- Age and gender estimation
- Facial landmarks
- Face attributes
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """
    Comprehensive facial analysis system
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize face analyzer
        
        Args:
            device: Device to run on
        """
        self.device = device
        
        # Emotion labels
        self.emotion_labels = [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ]
        
        # Face detector (using OpenCV DNN)
        self.face_detector = None
        self.load_face_detector()
        
        # Try to load DeepFace
        self.deepface_available = False
        try:
            import deepface
            self.deepface_available = True
            logger.info("DeepFace available for facial analysis")
        except ImportError:
            logger.warning("DeepFace not installed. Limited facial features.")
    
    def load_face_detector(self):
        """Load face detection model"""
        try:
            # Using OpenCV's DNN face detector
            modelFile = "opencv_face_detector_uint8.pb"
            configFile = "opencv_face_detector.pbtxt"
            
            # Fallback to Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("Face detector loaded (Haar Cascade)")
            
        except Exception as e:
            logger.warning(f"Face detector loading failed: {e}")
    
    def detect_faces(self, frame: np.ndarray, min_confidence: float = 0.5) -> List[Dict]:
        """
        Detect faces in frame
        
        Args:
            frame: Input image (BGR)
            min_confidence: Minimum detection confidence
            
        Returns:
            List of face detections
        """
        faces = []
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in detected:
                faces.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 1.0  # Haar cascade doesn't provide confidence
                })
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
        
        return faces
    
    def analyze_face(self, frame: np.ndarray, face_bbox: List[int]) -> Dict:
        """
        Comprehensive analysis of a single face
        
        Args:
            frame: Input image (BGR)
            face_bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with face analysis results
        """
        analysis = {
            'emotion': None,
            'age': None,
            'gender': None,
            'race': None,
            'landmarks': None
        }
        
        if not self.deepface_available:
            return analysis
        
        try:
            from deepface import DeepFace
            
            # Extract face region
            x1, y1, x2, y2 = face_bbox
            face_img = frame[y1:y2, x1:x2]
            
            # Analyze face
            result = DeepFace.analyze(
                face_img,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            analysis['emotion'] = result.get('dominant_emotion')
            analysis['age'] = result.get('age')
            analysis['gender'] = result.get('dominant_gender')
            analysis['race'] = result.get('dominant_race')
            
        except Exception as e:
            logger.error(f"Face analysis error: {e}")
        
        return analysis
    
    def get_facial_landmarks(self, frame: np.ndarray, face_bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract facial landmarks
        
        Args:
            frame: Input image
            face_bbox: Face bounding box
            
        Returns:
            Landmark coordinates
        """
        try:
            import dlib
            
            # Load dlib's face landmark predictor
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = face_bbox
            rect = dlib.rectangle(x1, y1, x2, y2)
            
            landmarks = predictor(gray, rect)
            coords = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            return coords
            
        except ImportError:
            logger.warning("dlib not available for landmark detection")
            return None
        except Exception as e:
            logger.error(f"Landmark detection error: {e}")
            return None
    
    def recognize_face(self, frame: np.ndarray, face_bbox: List[int], 
                      database_path: str = "face_db") -> Optional[str]:
        """
        Face recognition against database
        
        Args:
            frame: Input image
            face_bbox: Face bounding box
            database_path: Path to face database
            
        Returns:
            Recognized person name or None
        """
        if not self.deepface_available:
            return None
        
        try:
            from deepface import DeepFace
            
            x1, y1, x2, y2 = face_bbox
            face_img = frame[y1:y2, x1:x2]
            
            # Find matching face
            result = DeepFace.find(
                face_img,
                db_path=database_path,
                enforce_detection=False,
                silent=True
            )
            
            if len(result) > 0 and len(result[0]) > 0:
                # Return identity of best match
                return result[0].iloc[0]['identity']
            
        except Exception as e:
            logger.debug(f"Face recognition: {e}")
        
        return None
    
    def visualize_faces(self, frame: np.ndarray, faces: List[Dict], 
                       analyses: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Visualize face detections and analysis
        
        Args:
            frame: Input image
            faces: List of face detections
            analyses: Optional list of face analyses
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face['bbox'])
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add analysis info if available
            if analyses and i < len(analyses):
                analysis = analyses[i]
                y_offset = y1 - 10
                
                # Emotion
                if analysis.get('emotion'):
                    text = f"Emotion: {analysis['emotion']}"
                    cv2.putText(vis_frame, text, (x1, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_offset -= 20
                
                # Age and Gender
                if analysis.get('age') and analysis.get('gender'):
                    text = f"{analysis['gender']}, {analysis['age']}"
                    cv2.putText(vis_frame, text, (x1, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame


class FacialExpressionRecognizer:
    """
    Real-time facial expression recognition
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize expression recognizer
        
        Args:
            model_path: Optional path to custom model
        """
        self.model = None
        self.emotion_labels = [
            "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
        ]
    
    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Predict emotion from face image
        
        Args:
            face_img: Face image (grayscale or BGR)
            
        Returns:
            (emotion_label, confidence)
        """
        # Placeholder for emotion prediction
        # In production, would use trained CNN model
        return "Neutral", 0.8


class FaceTracker:
    """
    Multi-face tracking across frames
    """
    
    def __init__(self, max_disappeared: int = 30):
        """
        Initialize face tracker
        
        Args:
            max_disappeared: Maximum frames before removing track
        """
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid: np.ndarray) -> int:
        """Register new face track"""
        face_id = self.next_id
        self.objects[face_id] = centroid
        self.disappeared[face_id] = 0
        self.next_id += 1
        return face_id
    
    def deregister(self, face_id: int):
        """Remove face track"""
        del self.objects[face_id]
        del self.disappeared[face_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, np.ndarray]:
        """
        Update face tracks
        
        Args:
            detections: List of face detections
            
        Returns:
            Dictionary of face_id -> centroid
        """
        if len(detections) == 0:
            # Mark all as disappeared
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            return self.objects
        
        # Calculate centroids
        centroids = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroids.append(np.array([cx, cy]))
        
        centroids = np.array(centroids)
        
        # Register new faces if no existing tracks
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            # Match detections to existing tracks
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Simple nearest neighbor matching
            # In production, would use more sophisticated tracking
            for centroid in centroids:
                if len(object_centroids) > 0:
                    distances = [np.linalg.norm(centroid - oc) for oc in object_centroids]
                    min_idx = np.argmin(distances)
                    
                    if distances[min_idx] < 50:  # Threshold
                        face_id = object_ids[min_idx]
                        self.objects[face_id] = centroid
                        self.disappeared[face_id] = 0
                        object_ids.pop(min_idx)
                        object_centroids.pop(min_idx)
                    else:
                        self.register(centroid)
                else:
                    self.register(centroid)
        
        return self.objects

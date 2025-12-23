# Advanced Computer Vision System - Examples

This directory contains example scripts demonstrating various features of the CV system.

## Example Scripts

### 1. basic_detection.py
Basic object detection and tracking example.

```bash
python examples/basic_detection.py
```

### 2. pose_tracking.py
Human pose estimation and tracking example.

```bash
python examples/pose_tracking.py
```

### 3. face_analysis.py
Comprehensive facial analysis including emotion recognition.

```bash
python examples/face_analysis.py
```

### 4. depth_visualization.py
Monocular depth estimation visualization.

```bash
python examples/depth_visualization.py
```

### 5. scene_understanding.py
Scene classification and understanding using CLIP.

```bash
python examples/scene_understanding.py
```

## Custom Integration

To integrate specific models into your own application:

```python
from cv_system.camera import CameraInterface
from cv_system.models import ObjectDetector

# Initialize camera
camera = CameraInterface(camera_id=0)
camera.open()

# Initialize detector
detector = ObjectDetector(model_type="yolov8n")

# Process frames
while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Detect objects
    detections = detector.detect(frame)
    
    # Visualize
    vis_frame = detector.visualize(frame, detections)
    
    # Display or process...
```

## Batch Processing

Process video files:

```python
import cv2
from cv_system.models import ObjectDetector

detector = ObjectDetector(model_type="yolov8n")
cap = cv2.VideoCapture("input_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = detector.detect(frame)
    # Process detections...

cap.release()
```

## Performance Tips

1. **Model Selection**: Use smaller models (yolov8n) for real-time, larger (yolov8x) for accuracy
2. **Resolution**: Lower resolution = faster processing
3. **Feature Selection**: Disable unused features to improve FPS
4. **GPU**: Use CUDA-enabled GPU for 3-5x performance boost
5. **Batch Processing**: Process multiple frames together when possible

# Models Documentation

This document provides detailed information about the models used in the system.

## Object Detection

### YOLO v8/v9 (Ultralytics)
- **Papers**: YOLOv8 (2023), YOLOv9 (2024)
- **Purpose**: Real-time object detection
- **Classes**: 80 COCO classes (person, car, dog, etc.)
- **Variants**:
  - `yolov8n`: 3.2M params, fastest
  - `yolov8s`: 11.2M params
  - `yolov8m`: 25.9M params
  - `yolov8l`: 43.7M params
  - `yolov8x`: 68.2M params, most accurate

### Performance Benchmarks
| Model | mAP | Speed (ms) | Params |
|-------|-----|-----------|---------|
| YOLOv8n | 37.3 | 1.2 | 3.2M |
| YOLOv8s | 44.9 | 2.4 | 11.2M |
| YOLOv8m | 50.2 | 5.9 | 25.9M |
| YOLOv8l | 52.9 | 9.5 | 43.7M |
| YOLOv8x | 53.9 | 14.3 | 68.2M |

## Segmentation

### SAM (Segment Anything Model)
- **Paper**: ICCV 2023
- **Authors**: Meta AI
- **Purpose**: Universal image segmentation
- **Modes**:
  - Automatic mask generation
  - Point-prompted segmentation
  - Box-prompted segmentation
  - Text-prompted segmentation (with Grounding DINO)

### SegFormer
- **Paper**: NeurIPS 2021
- **Purpose**: Semantic segmentation
- **Architecture**: Vision Transformer-based
- **Datasets**: ADE20K, Cityscapes

## Depth Estimation

### DPT (Dense Prediction Transformer)
- **Paper**: ICCV 2021
- **Authors**: Intel ISL
- **Architecture**: Vision Transformer for dense prediction
- **Output**: Relative depth map

### MiDaS v3
- **Paper**: TPAMI 2021
- **Purpose**: Robust monocular depth estimation
- **Training**: Multiple diverse datasets
- **Output**: Normalized depth map (0-1)

## Pose Estimation

### MediaPipe Pose
- **Authors**: Google
- **Keypoints**: 33 landmarks (full body)
- **Speed**: Real-time on mobile devices
- **Accuracy**: High precision for visible joints

### ViTPose
- **Paper**: ECCV 2022
- **Architecture**: Vision Transformer for pose
- **Keypoints**: 17 COCO keypoints
- **Performance**: State-of-the-art accuracy

## Vision-Language Models

### CLIP
- **Paper**: ICML 2021
- **Authors**: OpenAI
- **Purpose**: Zero-shot image classification
- **Training**: 400M image-text pairs
- **Use Cases**:
  - Image classification with text prompts
  - Scene understanding
  - Image retrieval

### BLIP-2
- **Paper**: ICML 2023
- **Authors**: Salesforce
- **Purpose**: Image captioning, VQA
- **Architecture**: Q-Former + LLM
- **Capabilities**:
  - Natural language image descriptions
  - Visual question answering
  - Instruction-following

## Face Analysis

### DeepFace
- **Framework**: Comprehensive facial analysis
- **Features**:
  - Face detection
  - Age estimation
  - Gender classification
  - Emotion recognition (7 emotions)
  - Race estimation

### Face Recognition
- **Models**: VGG-Face, FaceNet, ArcFace
- **Use Cases**: Identity verification, face matching
- **Accuracy**: 99%+ on LFW benchmark

## Tracking

### SORT (Simple Online and Realtime Tracking)
- **Paper**: ICIP 2016
- **Method**: Kalman filter + Hungarian algorithm
- **Speed**: Real-time
- **Use**: Multi-object tracking

### DeepSORT
- **Paper**: 2017
- **Improvement**: Adds appearance features
- **Use**: More robust tracking with re-identification

## Model Selection Guide

### For Real-Time Applications (30+ FPS)
- Detection: YOLOv8n or YOLOv8s
- Pose: MediaPipe
- Depth: Disabled or MiDaS (small)

### For High Accuracy
- Detection: YOLOv8x
- Segmentation: SAM (ViT-H)
- Pose: ViTPose
- Depth: DPT-Large

### For Embedded/Mobile
- Detection: YOLOv8n (quantized)
- Pose: MediaPipe
- Face: Lightweight models only

## Fine-Tuning

Most models can be fine-tuned on custom datasets:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom data
model.train(data='custom_data.yaml', epochs=100)
```

## Model Downloads

Models are automatically downloaded on first use. Default locations:
- YOLO: `~/.cache/ultralytics/`
- Transformers: `~/.cache/huggingface/`
- Torch Hub: `~/.cache/torch/hub/`

## References

1. YOLOv8: https://github.com/ultralytics/ultralytics
2. SAM: https://segment-anything.com/
3. CLIP: https://github.com/openai/CLIP
4. BLIP-2: https://github.com/salesforce/LAVIS
5. MiDaS: https://github.com/isl-org/MiDaS
6. MediaPipe: https://google.github.io/mediapipe/

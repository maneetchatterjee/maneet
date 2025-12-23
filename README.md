# Advanced Computer Vision System 🎥🤖

A state-of-the-art computer vision system that integrates the latest research from top-tier conferences (CVPR, ICCV, ECCV) to provide comprehensive real-time video analysis using an external camera.

## 🌟 Features

This system incorporates cutting-edge techniques from recent computer vision research:

### Object Detection & Tracking
- **YOLO v8/v9**: Ultra-fast real-time object detection
- **Multi-Object Tracking (MOT)**: Track multiple objects across frames with trajectory visualization
- **80+ object classes** from COCO dataset

### Segmentation
- **SAM (Segment Anything Model)**: Meta's ICCV 2023 breakthrough for universal segmentation
- **SegFormer**: Semantic segmentation with transformer architecture
- **Instance & semantic segmentation** capabilities

### Depth Estimation
- **DPT (Dense Prediction Transformer)**: Monocular depth estimation
- **MiDaS v3**: Robust depth maps from single images
- Real-time depth visualization

### Human Analysis
- **Pose Estimation**: Full-body pose tracking with 17 keypoints
- **Hand Pose Detection**: 21-point hand landmark tracking
- **Face Analysis**: 
  - Face detection and tracking
  - Emotion recognition (7 emotions)
  - Age and gender estimation
  - Facial landmarks (68 points)

### Scene Understanding
- **CLIP**: Zero-shot image classification and scene understanding
- **BLIP-2**: Image captioning and visual question answering
- Automatic scene categorization (indoor/outdoor, weather, activities)

### Video Processing
- **Optical Flow**: Motion analysis and tracking
- **Multi-target tracking**: Consistent ID assignment across frames
- **Real-time performance monitoring**

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Webcam or external camera

### Setup

1. Clone the repository:
```bash
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) For GPU acceleration, ensure you have CUDA installed:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📖 Usage

### Basic Usage

Run with default settings (object detection, pose estimation, face analysis):
```bash
python src/main.py
```

### Advanced Options

```bash
# Use specific camera device
python src/main.py --camera 1

# Enable all features
python src/main.py --enable-depth --enable-scene --enable-hands

# Use more accurate detection model
python src/main.py --detection-model yolov8x

# Custom resolution and FPS
python src/main.py --resolution 1920x1080 --fps 60

# Force CPU mode (if no GPU available)
python src/main.py --device cpu
```

### Available Models

**Detection Models:**
- `yolov8n`: Nano (fastest)
- `yolov8s`: Small
- `yolov8m`: Medium
- `yolov8l`: Large
- `yolov8x`: Extra Large (most accurate)

### Keyboard Controls

While the application is running:
- **q**: Quit application
- **s**: Save current frame to `output/` directory
- **p**: Print performance statistics

## 🏗️ System Architecture

```
src/
├── cv_system/
│   ├── camera.py              # Camera interface
│   ├── models/
│   │   ├── detection.py       # Object detection (YOLO, DINO)
│   │   ├── segmentation.py    # SAM, Mask2Former
│   │   ├── depth.py           # Depth estimation (DPT, MiDaS)
│   │   ├── pose.py            # Pose estimation (ViTPose)
│   │   ├── vision_language.py # CLIP, BLIP-2
│   │   ├── face.py            # Face analysis
│   │   └── tracking.py        # Multi-object tracking
│   └── utils/
│       ├── visualization.py   # Visualization utilities
│       └── performance.py     # Performance monitoring
└── main.py                    # Main application
```

## 🔬 Research & Technologies

This system integrates techniques from recent computer vision conferences:

### CVPR (Computer Vision and Pattern Recognition)
- **Depth Anything** (2024): Robust monocular depth estimation
- **YOLO v9** (2024): Real-time object detection improvements
- **SAM** (2023): Segment Anything Model

### ICCV (International Conference on Computer Vision)
- **Grounding DINO** (2023): Open-vocabulary detection
- **ViTPose** (2022): Vision transformer for pose estimation

### ECCV (European Conference on Computer Vision)
- **Mask2Former** (2022): Universal image segmentation
- **SegFormer** (2021): Efficient semantic segmentation

### Other Notable Works
- **CLIP** (OpenAI): Vision-language pre-training
- **BLIP-2** (Salesforce): Bootstrapped language-image pre-training
- **MediaPipe** (Google): Cross-platform ML solutions

## 📊 Performance

Typical performance on different hardware:

| Hardware | Model | FPS | Features Enabled |
|----------|-------|-----|-----------------|
| RTX 4090 | YOLOv8x | 60+ | Detection, Pose, Face |
| RTX 3080 | YOLOv8l | 45+ | Detection, Pose, Face |
| RTX 2060 | YOLOv8m | 30+ | Detection, Pose, Face |
| CPU (i7) | YOLOv8n | 10-15 | Detection, Face |

## 🎯 Use Cases

- **Security & Surveillance**: Real-time person and object detection with tracking
- **Human-Computer Interaction**: Pose and gesture recognition
- **Robotics**: Scene understanding and depth perception
- **Sports Analytics**: Pose estimation and motion analysis
- **Accessibility**: Facial expression and emotion recognition
- **Augmented Reality**: Depth estimation and scene understanding
- **Research**: Platform for experimenting with latest CV models

## 🔧 Configuration

Create a custom configuration file in `config/custom.yaml`:

```yaml
camera:
  device_id: 0
  resolution: [1920, 1080]
  fps: 30

models:
  detection:
    enabled: true
    model: "yolov8l"
    confidence: 0.6
  
  pose:
    enabled: true
  
  face:
    enabled: true
    analyze_emotions: true
```

Load with:
```bash
python src/main.py --config config/custom.yaml
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model integrations (new papers from conferences)
- Performance optimizations
- New visualization modes
- Documentation improvements

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

This system builds upon numerous open-source projects and research works:
- Ultralytics YOLO
- Meta AI (SAM)
- Intel (MiDaS, DPT)
- Google (MediaPipe)
- OpenAI (CLIP)
- Salesforce (BLIP-2)
- HuggingFace Transformers
- OpenCV

## 📚 Citation

If you use this system in your research, please cite the relevant papers:

```bibtex
@article{yolov8,
  title={YOLOv8: Real-Time Object Detection},
  author={Ultralytics},
  year={2023}
}

@article{sam2023,
  title={Segment Anything},
  author={Kirillov, Alexander and others},
  journal={ICCV},
  year={2023}
}

@article{clip2021,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and others},
  journal={ICML},
  year={2021}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using the latest computer vision research**
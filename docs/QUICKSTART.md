# Quick Start Guide

Get up and running with the Advanced Computer Vision System in 5 minutes!

## 🚀 5-Minute Quickstart

### Step 1: Prerequisites Check

Make sure you have:
- ✅ Python 3.8 or higher
- ✅ A webcam or external camera
- ✅ (Optional) NVIDIA GPU with CUDA for faster processing

Check Python version:
```bash
python --version
# Should show Python 3.8.x or higher
```

### Step 2: Installation

**Option A: Automatic Setup (Recommended)**

Linux/Mac:
```bash
./setup.sh
source venv/bin/activate
```

Windows:
```bash
setup.bat
venv\Scripts\activate
```

**Option B: Manual Setup**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: First Run

Start with basic features:
```bash
python src/main.py
```

You should see:
1. Models loading (first time will download ~500MB)
2. Camera initializing
3. Real-time video with:
   - Object detection boxes
   - Person poses
   - Face detection
   - FPS counter

### Step 4: Test Features

Try these keyboard shortcuts:
- Press **'s'** - Save current frame to `output/` folder
- Press **'p'** - Print performance statistics
- Press **'q'** - Quit application

### Step 5: Explore More Features

Enable additional features:
```bash
# Add depth estimation
python src/main.py --enable-depth

# Add hand tracking
python src/main.py --enable-hands

# Use higher accuracy model
python src/main.py --detection-model yolov8l
```

## 🎯 Common Use Cases

### Security Camera
```bash
python src/main.py --detection-model yolov8m
```
- Detects people and vehicles
- Tracks multiple objects
- Shows trajectories

### Fitness Tracking
```bash
python src/main.py --no-detection
```
- Focus on pose estimation
- Track body movements
- Calculate joint angles

### Facial Analysis
```bash
python src/main.py --no-detection --no-pose
```
- Face detection and tracking
- Emotion recognition
- Age/gender estimation

### Full Feature Demo
```bash
python src/main.py --enable-depth --enable-hands --enable-scene
```
- Everything enabled!
- Requires powerful GPU
- Lower FPS but maximum features

## 📊 Expected Performance

| Setup | FPS | Features |
|-------|-----|----------|
| Basic (CPU) | 10-15 | Detection + Face |
| Standard (GPU) | 30+ | Detection + Pose + Face |
| Full (High-end GPU) | 20-30 | All features |

## 🎨 Customization

### Change Resolution
```bash
python src/main.py --resolution 1920x1080
```

### Select Different Camera
```bash
# List available cameras (Linux)
v4l2-ctl --list-devices

# Use camera 1
python src/main.py --camera 1
```

### Adjust Confidence Threshold
Edit `src/main.py` line with `conf_threshold=0.5` to your desired value (0.1-0.9)

## 🔍 Verify Installation

Test individual components:

```python
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED'); cap.release()"

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test YOLO
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO OK')"
```

## 📱 Next Steps

1. **Read the full documentation**: `README.md`
2. **Explore examples**: `docs/EXAMPLES.md`
3. **Learn about models**: `docs/MODELS.md`
4. **Troubleshooting**: `docs/TROUBLESHOOTING.md`

## 🆘 Getting Help

If something doesn't work:
1. Check `docs/TROUBLESHOOTING.md`
2. Ensure all dependencies installed: `pip list`
3. Check camera access: Test with other applications
4. Open an issue on GitHub with:
   - Error message
   - Python version
   - OS and hardware
   - Steps to reproduce

## 🎉 Success!

If you see real-time video with bounding boxes and poses, you're all set! 

Enjoy exploring state-of-the-art computer vision! 🚀

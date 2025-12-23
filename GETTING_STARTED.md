# 🚀 Getting Started with Advanced Computer Vision System

This guide will help you get the system running in just a few minutes!

## ⚡ Quick Setup (3 Steps)

### Step 1: Clone and Navigate
```bash
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet
```

### Step 2: Install Dependencies

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows:**
```batch
setup.bat
venv\Scripts\activate
```

### Step 3: Run!
```bash
python src/main.py
```

That's it! Your camera should open with real-time CV features! 🎉

## 📹 What You'll See

When you run the application, you'll see:

1. **Object Detection Boxes** - Green boxes around detected objects (people, cars, etc.)
2. **Tracking IDs** - Persistent IDs for tracked objects with trajectory lines
3. **Pose Skeletons** - Blue lines showing body pose of detected people
4. **Face Boxes** - Green rectangles around detected faces
5. **FPS Counter** - Real-time frames per second in top-right
6. **Info Panel** - Object counts and statistics in top-left

## ⌨️ Controls

While running:
- **q** - Quit application
- **s** - Save current frame to `output/` folder
- **p** - Print detailed performance statistics

## 🎮 Try Different Modes

### Basic Mode (Fast)
```bash
python src/main.py
```
Default: Object detection, pose estimation, face detection

### High Accuracy Mode
```bash
python src/main.py --detection-model yolov8x
```
Uses largest, most accurate model (slower but better)

### All Features Enabled
```bash
python src/main.py --enable-depth --enable-hands --enable-scene
```
Enables depth estimation, hand tracking, and scene understanding

### CPU Mode (No GPU)
```bash
python src/main.py --device cpu --detection-model yolov8n
```
Runs on CPU with fastest model

### Custom Camera
```bash
python src/main.py --camera 1
```
Use camera device 1 instead of default (0)

### High Resolution
```bash
python src/main.py --resolution 1920x1080
```
Run at 1080p resolution

## 🔍 Validate Installation

Before running with camera, validate your setup:

```bash
python src/validate.py
```

This checks:
- ✓ All modules can be imported
- ✓ Dependencies are installed
- ✓ CUDA is available (if GPU present)
- ✓ Basic functionality works

## 📊 Expected Performance

| Your Hardware | Recommended Command | Expected FPS |
|---------------|-------------------|--------------|
| RTX 3080/4090 | `python src/main.py --detection-model yolov8l` | 45-60 |
| RTX 2060/3060 | `python src/main.py` | 30-40 |
| GTX 1660 | `python src/main.py --detection-model yolov8n` | 25-35 |
| No GPU (CPU) | `python src/main.py --device cpu --no-pose` | 5-15 |

## 🆘 Common Issues

### Issue: Camera not opening
**Solution:**
```bash
# Check available cameras
ls /dev/video*  # Linux
# or try different camera ID
python src/main.py --camera 1
```

### Issue: Out of memory
**Solution:**
```bash
# Use smaller model
python src/main.py --detection-model yolov8n
```

### Issue: ModuleNotFoundError
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Slow performance
**Solution:**
```bash
# Reduce resolution and use smaller model
python src/main.py --resolution 640x480 --detection-model yolov8n
```

See `docs/TROUBLESHOOTING.md` for more solutions!

## 📚 Learn More

- **Full Documentation**: `README.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Quick Start Guide**: `docs/QUICKSTART.md`
- **Usage Examples**: `docs/EXAMPLES.md`
- **Model Information**: `docs/MODELS.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

## 🎯 Next Steps

1. ✅ Run basic detection: `python src/main.py`
2. 📸 Save some frames with 's' key
3. 🔧 Try different models and settings
4. 📖 Read `docs/EXAMPLES.md` for advanced usage
5. 🚀 Integrate into your own project!

## 💡 Tips

- **Start simple**: Use default settings first
- **Monitor FPS**: Use 'p' key to see performance stats
- **Adjust model size**: Balance speed vs accuracy for your needs
- **Good lighting**: Better lighting = better results
- **Stable camera**: Mount camera for best tracking results

## 🎉 Have Fun!

This system integrates cutting-edge research from CVPR, ICCV, and ECCV. Experiment with different features and see what works best for your application!

**Questions?** Check `docs/TROUBLESHOOTING.md` or open an issue on GitHub.

---

**Happy Computer Visioning!** 🤖📷✨

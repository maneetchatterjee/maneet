# Troubleshooting Guide

Common issues and solutions for the Advanced Computer Vision System.

## 🔴 Installation Issues

### Issue: `pip install` fails with compilation errors

**Solution**:
```bash
# Upgrade pip and wheel
pip install --upgrade pip wheel setuptools

# Install with pre-built binaries
pip install --no-cache-dir -r requirements.txt

# If still failing, install problematic packages separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CUDA not detected despite having NVIDIA GPU

**Check CUDA installation**:
```bash
nvidia-smi  # Should show GPU and CUDA version
```

**Install correct PyTorch version**:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: `ModuleNotFoundError` when running

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Reinstall requirements
pip install -r requirements.txt

# Check installation
pip list | grep torch
pip list | grep opencv
```

## 📷 Camera Issues

### Issue: "Failed to open camera"

**Solutions**:

1. **Check camera availability**:
```bash
# Linux
ls /dev/video*
v4l2-ctl --list-devices

# Test with simple script
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAILED')"
```

2. **Try different camera IDs**:
```bash
python src/main.py --camera 0
python src/main.py --camera 1
python src/main.py --camera 2
```

3. **Check permissions** (Linux):
```bash
sudo usermod -a -G video $USER
# Logout and login again
```

4. **Camera in use by another application**:
- Close other apps using camera (Zoom, Teams, etc.)
- Restart computer

### Issue: Low FPS or laggy video

**Solutions**:

1. **Use lighter model**:
```bash
python src/main.py --detection-model yolov8n
```

2. **Reduce resolution**:
```bash
python src/main.py --resolution 640x480
```

3. **Disable heavy features**:
```bash
python src/main.py --no-pose
```

4. **Force CPU** (if GPU is slow):
```bash
python src/main.py --device cpu
```

## 🧠 Model Issues

### Issue: "Model not found" or download errors

**Solutions**:

1. **Manual download**:
```bash
# Create models directory
mkdir -p ~/.cache/ultralytics

# Download manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P ~/.cache/ultralytics/
```

2. **Check internet connection**:
```bash
ping github.com
```

3. **Use different mirror** (for HuggingFace models):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Issue: Out of memory (OOM) errors

**Solutions**:

1. **Use smaller model**:
```bash
python src/main.py --detection-model yolov8n
```

2. **Reduce batch size** (edit code):
```python
# In detection.py, process frames individually
```

3. **Close other applications**:
- Free up GPU/RAM
- Monitor usage: `nvidia-smi` (GPU) or `htop` (RAM)

4. **Enable mixed precision**:
```python
# In model loading, add:
model.half()  # Use FP16 instead of FP32
```

## 🖥️ Performance Issues

### Issue: Very low FPS (< 5)

**Diagnostic**:
```bash
# Run with performance monitoring
python src/main.py
# Press 'p' to see which component is slow
```

**Solutions based on bottleneck**:

1. **Detection slow**: Use yolov8n
2. **Depth slow**: Disable with default config
3. **Segmentation slow**: Keep disabled by default
4. **CPU bottleneck**: Use GPU or reduce resolution
5. **GPU bottleneck**: Use smaller models

### Issue: High CPU usage

**Solutions**:
```python
# In main.py, limit frame processing
import time
time.sleep(0.01)  # Add small delay between frames
```

### Issue: Memory leak (RAM increases over time)

**Solutions**:
```python
# Explicit garbage collection
import gc
gc.collect()
torch.cuda.empty_cache()  # For GPU memory
```

## 🎨 Visualization Issues

### Issue: Window doesn't appear

**Solutions**:

1. **Check display** (Linux):
```bash
echo $DISPLAY
# Should show :0 or similar
```

2. **Enable X forwarding** (SSH):
```bash
ssh -X user@host
```

3. **Use headless mode** (add to code):
```python
# Save to file instead of display
cv2.imwrite('output.jpg', frame)
```

### Issue: Black screen or corrupted display

**Solutions**:

1. **Update OpenCV**:
```bash
pip install --upgrade opencv-python
```

2. **Try different backend**:
```python
cv2.namedWindow('window', cv2.WINDOW_NORMAL)
```

## 🐧 Platform-Specific Issues

### Linux

**Issue: Permission denied for camera**
```bash
sudo chmod 666 /dev/video0
# Or permanently:
sudo usermod -a -G video $USER
```

**Issue: Missing system dependencies**
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libopencv-dev
```

### Windows

**Issue: DLL load failed**
```bash
# Install Visual C++ Redistributable
# Download from Microsoft website
```

**Issue: Long paths**
```bash
# Enable long paths in Windows
# Run PowerShell as admin:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### macOS

**Issue: Camera permission denied**
```bash
# System Preferences > Security & Privacy > Camera
# Enable for Terminal/iTerm
```

**Issue: Metal/GPU not available**
```bash
# Install PyTorch with MPS support
pip install torch torchvision
```

## 📋 Debug Mode

Enable verbose logging:

```python
# In main.py, set logging level
logging.basicConfig(level=logging.DEBUG)
```

## 🆘 Still Having Issues?

1. **Collect information**:
```bash
python --version
pip list
nvidia-smi  # If applicable
uname -a  # Linux/Mac
```

2. **Create minimal reproduction**:
```python
# Test basic functionality
import cv2
import torch
from ultralytics import YOLO

print(f"OpenCV: {cv2.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

cap = cv2.VideoCapture(0)
print(f"Camera: {cap.isOpened()}")
cap.release()
```

3. **Open GitHub issue** with:
   - Error message (full traceback)
   - System information
   - Steps to reproduce
   - What you've tried

## 💡 Tips for Best Performance

1. **Always use GPU** when available
2. **Start simple**: Enable features one at a time
3. **Match model size** to your hardware
4. **Close background applications**
5. **Use appropriate resolution** (720p is usually good)
6. **Update drivers**: GPU, camera, system

## 📚 Additional Resources

- [PyTorch Troubleshooting](https://pytorch.org/get-started/locally/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

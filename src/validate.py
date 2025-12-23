#!/usr/bin/env python3
"""
Validation script to test basic system components without requiring a camera
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from cv_system import CameraInterface
        print("✓ Camera interface imported")
    except Exception as e:
        print(f"✗ Camera interface failed: {e}")
        return False
    
    try:
        from cv_system.models import (
            ObjectDetector,
            SegmentAnything,
            SemanticSegmentation,
            DepthEstimator,
            PoseEstimator,
            HandPoseEstimator,
            CLIPModel,
            BLIP2,
            SceneUnderstanding,
            FaceAnalyzer,
            MultiObjectTracker,
            SingleObjectTracker,
            OpticalFlowTracker,
        )
        print("✓ All model modules imported")
    except Exception as e:
        print(f"✗ Model imports failed: {e}")
        return False
    
    try:
        from cv_system.utils import (
            Visualizer,
            FPSCounter,
            PerformanceMonitor,
            FrameBuffer,
        )
        print("✓ All utility modules imported")
    except Exception as e:
        print(f"✗ Utility imports failed: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that core dependencies are available"""
    print("\nTesting dependencies...")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO")
    except ImportError:
        print("⚠ Ultralytics not installed (object detection will not work)")
    
    try:
        from transformers import __version__ as tf_version
        print(f"✓ Transformers {tf_version}")
    except ImportError:
        print("⚠ Transformers not installed (some models will not work)")
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe {mp.__version__}")
    except ImportError:
        print("⚠ MediaPipe not installed (pose estimation will have limited features)")
    
    return True


def test_basic_functionality():
    """Test basic functionality without camera"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from cv_system.utils import FPSCounter, PerformanceMonitor, Visualizer
        
        # Test FPS counter
        fps_counter = FPSCounter()
        for _ in range(10):
            fps = fps_counter.tick()
        print(f"✓ FPS Counter working (current: {fps:.1f})")
        
        # Test performance monitor
        perf = PerformanceMonitor()
        perf.start_timer("test")
        import time
        time.sleep(0.01)
        perf.stop_timer("test")
        avg_time = perf.get_avg_time("test")
        print(f"✓ Performance Monitor working (test time: {avg_time*1000:.2f}ms)")
        
        # Test visualizer
        vis = Visualizer()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = {"Test": "Value", "Number": 42}
        vis_frame = vis.draw_info_panel(dummy_frame, info)
        print(f"✓ Visualizer working (frame shape: {vis_frame.shape})")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    return True


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("Advanced Computer Vision System - Validation")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_dependencies():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n✅ System validation successful!")
        print("\nYou can now run the main application:")
        print("  python src/main.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Activate virtual environment: source venv/bin/activate")
        print("  - Check Python version: python --version (need 3.8+)")
        return 1


if __name__ == '__main__':
    sys.exit(main())

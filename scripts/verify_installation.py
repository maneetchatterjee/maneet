#!/usr/bin/env python3
"""
Verify VLA pipeline installation and dependencies.
Usage: python scripts/verify_installation.py
"""

import sys
import importlib.util

def check_package(package_name, display_name=None):
    """Check if a package is installed."""
    if display_name is None:
        display_name = package_name
    
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ {display_name} - NOT INSTALLED")
        return False
    else:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name} - {version}")
            return True
        except ImportError as e:
            print(f"❌ {display_name} - IMPORT ERROR: {e}")
            return False

def main():
    print("=" * 50)
    print("VLA Pipeline Installation Verification")
    print("=" * 50)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    print()
    
    # Check required packages
    print("Checking required packages...")
    packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('cv2', 'OpenCV'),
        ('pybullet', 'PyBullet'),
        ('sklearn', 'scikit-learn'),
        ('networkx', 'NetworkX')
    ]
    
    all_ok = True
    for pkg, name in packages:
        if not check_package(pkg, name):
            all_ok = False
    
    print()
    
    if all_ok:
        print("=" * 50)
        print("✓ All dependencies verified successfully!")
        print("=" * 50)
        print()
        print("You can now run:")
        print("  python demo/demo_basic.py --seed 42")
        return True
    else:
        print("=" * 50)
        print("❌ Some dependencies are missing")
        print("=" * 50)
        print()
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

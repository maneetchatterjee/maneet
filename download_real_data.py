"""
Download the NASA SMAP-MSL dataset from Kaggle.

This script automates the download of the real NASA SMAP-MSL anomaly detection dataset.

Prerequisites:
1. A Kaggle account (sign up at https://www.kaggle.com)
2. Kaggle API credentials

Setup Instructions:
1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads kaggle.json
5. Place it in ~/.kaggle/kaggle.json (or set KAGGLE_CONFIG_DIR environment variable)
6. Run this script: python download_real_data.py
"""

import os
import sys
import subprocess
import zipfile
import shutil


def check_kaggle_setup():
    """Check if Kaggle CLI and credentials are available."""
    # Check if kaggle is installed
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        print(f"✓ Kaggle CLI installed: {result.stdout.strip()}")
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], check=True)
        print("✓ Kaggle CLI installed")
    
    # Check for credentials
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("\n" + "="*70)
        print("✗ Kaggle API credentials not found!")
        print("="*70)
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This downloads kaggle.json")
        print(f"5. Move it to: {kaggle_json}")
        print(f"6. Set permissions: chmod 600 {kaggle_json}")
        print("7. Run this script again")
        print("="*70)
        return False
    
    print(f"✓ Kaggle credentials found at {kaggle_json}")
    
    # Check permissions
    if os.name != 'nt':  # Not Windows
        stat_info = os.stat(kaggle_json)
        if stat_info.st_mode & 0o077:
            print(f"  ⚠ Fixing permissions on {kaggle_json}")
            os.chmod(kaggle_json, 0o600)
    
    return True


def download_dataset(data_dir='./data'):
    """Download NASA SMAP-MSL dataset from Kaggle."""
    dataset_name = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
    
    print(f"\nDownloading dataset: {dataset_name}")
    print("This may take a few minutes depending on your connection...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download using Kaggle CLI
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', data_dir],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"✗ Download failed:")
            print(result.stderr)
            return False
        
        print("✓ Dataset downloaded")
        
        # Unzip
        zip_path = os.path.join(data_dir, 'nasa-anomaly-detection-dataset-smap-msl.zip')
        if not os.path.exists(zip_path):
            print(f"✗ Expected zip file not found: {zip_path}")
            return False
        
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("✓ Dataset extracted")
        
        # Reorganize directory structure if needed
        data_subdir = os.path.join(data_dir, 'data')
        if os.path.exists(data_subdir):
            print("Reorganizing directory structure...")
            for item in os.listdir(data_subdir):
                src = os.path.join(data_subdir, item)
                dst = os.path.join(data_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            shutil.rmtree(data_subdir)
            print("✓ Directory structure reorganized")
        
        # Clean up zip file
        os.remove(zip_path)
        print("✓ Cleaned up zip file")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Download timed out. Please try again or download manually.")
        return False
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False


def verify_dataset(data_dir='./data'):
    """Verify the downloaded dataset."""
    print("\nVerifying dataset...")
    
    required_dirs = ['train', 'test']
    required_file = 'labeled_anomalies.csv'
    
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"✗ Missing directory: {dir_name}")
            return False
        
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
        print(f"✓ {dir_name}/ directory found with {len(files)} .npy files")
    
    csv_path = os.path.join(data_dir, required_file)
    if not os.path.exists(csv_path):
        print(f"✗ Missing file: {required_file}")
        return False
    
    print(f"✓ {required_file} found")
    
    # Load and display info
    import pandas as pd
    labels_df = pd.read_csv(csv_path)
    print(f"\n✓ Dataset verified!")
    print(f"  - Total channels: {len(labels_df)}")
    print(f"  - SMAP channels: {len(labels_df[labels_df['spacecraft'] == 'SMAP'])}")
    print(f"  - MSL channels: {len(labels_df[labels_df['spacecraft'] == 'MSL'])}")
    
    return True


def main():
    """Main function."""
    print("="*70)
    print("NASA SMAP-MSL Dataset Downloader")
    print("="*70)
    
    # Check setup
    if not check_kaggle_setup():
        sys.exit(1)
    
    # Download dataset
    if not download_dataset():
        print("\n✗ Download failed. Please try manual download:")
        print("1. Visit: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
        print("2. Download the dataset")
        print("3. Extract to ./data/ directory")
        sys.exit(1)
    
    # Verify dataset
    if not verify_dataset():
        print("\n✗ Dataset verification failed")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ SUCCESS! NASA SMAP-MSL dataset is ready to use")
    print("="*70)
    print("\nYou can now run:")
    print("  python experiment.py")
    print("  python quick_test.py")
    print("="*70)


if __name__ == "__main__":
    main()

"""
Data loader for NASA SMAP-MSL anomaly detection dataset.
Based on the dataset from: https://github.com/khundman/telemanom
Dataset available at: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
"""

import os
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import pickle
import zipfile


class NASADataLoader:
    """Load and preprocess NASA SMAP-MSL anomaly detection dataset."""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_data_from_kaggle(self):
        """
        Download NASA SMAP-MSL dataset from Kaggle.
        
        Requires:
        1. pip install kaggle
        2. Kaggle API credentials in ~/.kaggle/kaggle.json
        
        Get your API key from: https://www.kaggle.com/account
        """
        print("Downloading NASA SMAP-MSL dataset from Kaggle...")
        
        try:
            import subprocess
            import zipfile
            
            # Download using Kaggle CLI
            dataset_name = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
            print(f"Downloading dataset: {dataset_name}")
            
            # Download to temp location
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', self.data_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"Error downloading from Kaggle:")
                print(result.stderr)
                raise Exception("Kaggle download failed. Please ensure you have Kaggle API credentials set up.")
            
            # Unzip the downloaded file
            zip_path = os.path.join(self.data_dir, 'nasa-anomaly-detection-dataset-smap-msl.zip')
            if os.path.exists(zip_path):
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # Reorganize files if needed
                data_subdir = os.path.join(self.data_dir, 'data')
                if os.path.exists(data_subdir):
                    # Move files up one level
                    import shutil
                    for item in os.listdir(data_subdir):
                        src = os.path.join(data_subdir, item)
                        dst = os.path.join(self.data_dir, item)
                        if os.path.exists(dst):
                            if os.path.isdir(dst):
                                shutil.rmtree(dst)
                            else:
                                os.remove(dst)
                        shutil.move(src, dst)
                    shutil.rmtree(data_subdir)
                
                os.remove(zip_path)
                print("✓ Dataset downloaded and extracted successfully!")
            else:
                raise Exception("Downloaded zip file not found")
                
        except Exception as e:
            print(f"\n✗ Error downloading from Kaggle: {e}")
            print("\nTo download the NASA SMAP-MSL dataset:")
            print("1. Install Kaggle CLI: pip install kaggle")
            print("2. Get your API key from https://www.kaggle.com/account")
            print("3. Place kaggle.json in ~/.kaggle/kaggle.json")
            print("4. Run: kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
            print("5. Extract to ./data/ directory")
            raise
    
    def download_data(self):
        """Download NASA SMAP-MSL dataset (legacy method - tries multiple sources)."""
        print("Attempting to download NASA SMAP-MSL dataset...")
        
        # Try Kaggle first (primary source)
        try:
            self.download_data_from_kaggle()
            return
        except Exception as e:
            print(f"Kaggle download failed: {e}")
        
        # Fallback: Try direct download (may not work)
        print("\nTrying direct download from GitHub (may not have data files)...")
        base_url = "https://raw.githubusercontent.com/khundman/telemanom/master"
        
        # Download labeled_anomalies.csv at least
        csv_url = f"{base_url}/labeled_anomalies.csv"
        csv_path = os.path.join(self.data_dir, 'labeled_anomalies.csv')
        
        try:
            response = requests.get(csv_url, timeout=30)
            if response.status_code == 200:
                with open(csv_path, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded: labeled_anomalies.csv")
            else:
                print(f"✗ Failed to download labeled_anomalies.csv")
        except Exception as e:
            print(f"✗ Error downloading labeled_anomalies.csv: {e}")
        
        print("\n" + "="*70)
        print("⚠ IMPORTANT: Data files not available from direct download")
        print("="*70)
        print("Please download the NASA SMAP-MSL dataset manually:")
        print("1. Go to: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
        print("2. Download the dataset")
        print("3. Extract to ./data/ directory")
        print("OR use Kaggle CLI:")
        print("  kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
        print("="*70)
    
    def load_data(self, channel='P-1', sequence_length=50):
        """
        Load and preprocess data for training.
        
        Args:
            channel: Channel name (e.g., 'P-1', 'S-1', 'E-1', 'M-1')
            sequence_length: Length of sequences for LSTM input
            
        Returns:
            X_train, y_train, X_test, y_test, labels
        """
        # Load training and test data
        train_path = os.path.join(self.data_dir, 'train', f'{channel}.npy')
        test_path = os.path.join(self.data_dir, 'test', f'{channel}.npy')
        
        if not os.path.exists(train_path):
            print("Data not found. Downloading...")
            self.download_data()
        
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        
        # Load anomaly labels
        labels_path = os.path.join(self.data_dir, 'labeled_anomalies.csv')
        labels_df = pd.read_csv(labels_path)
        
        # Get anomaly labels for this channel
        channel_labels = labels_df[labels_df['chan_id'] == channel]
        
        # Create binary labels for test data
        test_labels = np.zeros(len(test_data))
        for _, row in channel_labels.iterrows():
            anomaly_sequences = eval(row['anomaly_sequences'])
            for start, end in anomaly_sequences:
                test_labels[start:end+1] = 1
        
        # Normalize data
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0) + 1e-8
        
        train_data = (train_data - train_mean) / train_std
        test_data = (test_data - train_mean) / train_std
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data, sequence_length)
        X_test, y_test = self.create_sequences(test_data, sequence_length)
        
        # Adjust labels for sequences
        test_labels_seq = test_labels[sequence_length:]
        
        return X_train, y_train, X_test, y_test, test_labels_seq
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)
    
    def get_available_channels(self):
        """Get list of available channels."""
        labels_path = os.path.join(self.data_dir, 'labeled_anomalies.csv')
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            return labels_df['chan_id'].unique().tolist()
        return []


if __name__ == "__main__":
    # Test data loader
    loader = NASADataLoader()
    loader.download_data()
    
    print("\nAvailable channels:")
    print(loader.get_available_channels())
    
    print("\nLoading sample data for channel P-1...")
    X_train, y_train, X_test, y_test, labels = loader.load_data('P-1')
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {labels.shape}")
    print(f"Anomalies in test set: {np.sum(labels)} / {len(labels)}")

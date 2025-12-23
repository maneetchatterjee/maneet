"""
Data loader for NASA SMAP-MSL anomaly detection dataset.
Based on the dataset from: https://github.com/khundman/telemanom
"""

import os
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import pickle


class NASADataLoader:
    """Load and preprocess NASA SMAP-MSL anomaly detection dataset."""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_data(self):
        """Download NASA SMAP-MSL dataset."""
        print("Downloading NASA SMAP-MSL dataset...")
        
        # Base URL for the dataset
        base_url = "https://raw.githubusercontent.com/khundman/telemanom/master/data"
        
        files = [
            'train/P-1.npy',
            'train/S-1.npy',
            'train/E-1.npy',
            'train/M-1.npy',
            'test/P-1.npy',
            'test/S-1.npy',
            'test/E-1.npy',
            'test/M-1.npy',
            'labeled_anomalies.csv'
        ]
        
        for file_path in files:
            url = f"{base_url}/{file_path}"
            local_path = os.path.join(self.data_dir, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {file_path}")
                else:
                    print(f"Failed to download: {file_path}")
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")
    
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

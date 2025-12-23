"""
Generate synthetic NASA-like data for testing when real data is unavailable.
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_data(n_samples=5000, n_features=25, anomaly_ratio=0.03, seed=42):
    """
    Generate synthetic time series data similar to NASA SMAP-MSL.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features/sensors
        anomaly_ratio: Ratio of anomalous points
        seed: Random seed for reproducibility
        
    Returns:
        data: Synthetic time series data
        labels: Binary anomaly labels
    """
    np.random.seed(seed)
    
    # Generate normal data with autocorrelation
    t = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Add trend
        trend = 0.001 * t * np.sin(2 * np.pi * i / n_features)
        
        # Add seasonality
        seasonal = np.sin(2 * np.pi * t / (100 + 20 * i)) + np.cos(2 * np.pi * t / (200 + 30 * i))
        
        # Add noise
        noise = np.random.randn(n_samples) * 0.5
        
        # Combine components
        data[:, i] = trend + seasonal + noise
    
    # Generate anomalies
    labels = np.zeros(n_samples)
    n_anomalies = int(n_samples * anomaly_ratio)
    
    # Create anomaly windows
    anomaly_starts = np.random.choice(n_samples - 50, size=n_anomalies // 10, replace=False)
    
    for start in anomaly_starts:
        window_size = np.random.randint(5, 20)
        end = min(start + window_size, n_samples)
        
        # Mark as anomalies
        labels[start:end] = 1
        
        # Inject anomalies (spikes, shifts, or outliers)
        anomaly_type = np.random.choice(['spike', 'shift', 'noise'])
        
        if anomaly_type == 'spike':
            # Sudden spike
            data[start:end] += np.random.randn(end - start, n_features) * 5
        elif anomaly_type == 'shift':
            # Level shift
            data[start:end] += np.random.randn(1, n_features) * 3
        else:
            # High noise
            data[start:end] += np.random.randn(end - start, n_features) * 2
    
    return data, labels


def save_synthetic_dataset(data_dir='./data'):
    """Save synthetic dataset in NASA SMAP-MSL format."""
    os.makedirs(f"{data_dir}/train", exist_ok=True)
    os.makedirs(f"{data_dir}/test", exist_ok=True)
    
    print("Generating synthetic NASA-like dataset...")
    
    # Generate data for different channels
    channels = ['P-1', 'S-1', 'E-1', 'M-1']
    
    all_labels = []
    
    for channel in channels:
        print(f"Generating {channel}...")
        
        # Training data (no anomalies)
        train_data, _ = generate_synthetic_data(n_samples=8000, anomaly_ratio=0.0)
        np.save(f"{data_dir}/train/{channel}.npy", train_data)
        
        # Test data (with anomalies)
        test_data, test_labels = generate_synthetic_data(n_samples=3000, anomaly_ratio=0.05)
        np.save(f"{data_dir}/test/{channel}.npy", test_data)
        
        # Store label information
        # Find anomaly sequences
        anomaly_sequences = []
        in_anomaly = False
        start = None
        
        for i, label in enumerate(test_labels):
            if label == 1 and not in_anomaly:
                start = i
                in_anomaly = True
            elif label == 0 and in_anomaly:
                anomaly_sequences.append([start, i - 1])
                in_anomaly = False
        
        if in_anomaly:
            anomaly_sequences.append([start, len(test_labels) - 1])
        
        all_labels.append({
            'chan_id': channel,
            'spacecraft': 'synthetic',
            'anomaly_sequences': str(anomaly_sequences),
            'class': 'synthetic',
            'num_values': len(test_data)
        })
    
    # Save labels CSV
    labels_df = pd.DataFrame(all_labels)
    labels_df.to_csv(f"{data_dir}/labeled_anomalies.csv", index=False)
    
    print(f"\nSynthetic dataset saved to {data_dir}/")
    print(f"Channels: {channels}")
    print(f"Training samples per channel: 8000")
    print(f"Test samples per channel: 3000")
    print(f"Features per channel: 25")


if __name__ == "__main__":
    save_synthetic_dataset()
    print("\nSynthetic dataset generation complete!")

# Running Experiments on Real NASA SMAP-MSL Data

## Overview

The actual NASA SMAP-MSL dataset is required to run experiments on real spacecraft telemetry data. This guide explains how to obtain and use the real dataset.

## Current Status

⚠️ **The repository currently uses synthetic data** that mimics NASA spacecraft telemetry characteristics. To run experiments on the **actual NASA SMAP-MSL dataset**, you need to download it from Kaggle.

## Why Real Data is Important

The synthetic data was created to demonstrate the methodology when the real dataset download failed. However, for production use and accurate comparisons with published results, you should use the real NASA SMAP-MSL dataset:

- **Real Data**: 55 channels of actual spacecraft telemetry from SMAP and MSL missions
- **Synthetic Data**: 4 channels of generated time series that mimic spacecraft patterns

Expected performance differences:
- **Real Data**: F1-scores typically 0.60-0.80 (published results)
- **Synthetic Data**: F1-scores >0.95 (patterns are more regular)

## How to Download Real NASA SMAP-MSL Data

### Option 1: Automated Download (Recommended)

1. **Set up Kaggle API credentials:**
   ```bash
   # Go to https://www.kaggle.com/account
   # Click "Create New API Token"
   # This downloads kaggle.json
   # Move it to ~/.kaggle/kaggle.json
   
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Run the download script:**
   ```bash
   python download_real_data.py
   ```

   This will:
   - Download the dataset from Kaggle (~50MB)
   - Extract it to `./data/` directory
   - Verify the dataset structure
   - Display dataset statistics

### Option 2: Manual Download

1. **Visit the Kaggle dataset page:**
   https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

2. **Download the dataset:**
   - Click "Download" button (requires Kaggle account)
   - Wait for download to complete

3. **Extract to data directory:**
   ```bash
   # Extract the zip file
   unzip nasa-anomaly-detection-dataset-smap-msl.zip -d data/
   
   # Reorganize if needed
   if [ -d "data/data" ]; then
       mv data/data/* data/
       rmdir data/data
   fi
   ```

4. **Verify the structure:**
   ```bash
   ls data/
   # Should show: train/ test/ labeled_anomalies.csv
   ```

### Option 3: Using Kaggle CLI Directly

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (see Option 1)

# Download dataset
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl

# Extract
unzip nasa-anomaly-detection-dataset-smap-msl.zip -d data/

# Reorganize
mv data/data/* data/ && rmdir data/data
```

## Dataset Structure

After download, your `data/` directory should contain:

```
data/
├── train/
│   ├── P-1.npy     (SMAP channel 1)
│   ├── P-2.npy     (SMAP channel 2)
│   ├── ...         (25 SMAP channels total)
│   ├── S-1.npy     (MSL channel 1)
│   ├── ...         (27 MSL channels total)
│   ├── E-1.npy
│   ├── ...
│   └── M-1.npy
│   
├── test/
│   ├── P-1.npy
│   ├── ...
│   └── (same channels as train/)
│   
└── labeled_anomalies.csv   (anomaly labels for all channels)
```

**Total:** 55 channels (25 SMAP + 27 MSL + 3 others)

## Running Experiments with Real Data

Once you have the real data downloaded:

### Quick Test (Single Channel)
```bash
# Run on P-1 channel with 15 epochs (~15 minutes)
python quick_test.py
```

### Full Experiment (Single Channel)
```bash
# Run on P-1 channel with 30 epochs (~30 minutes)
python experiment.py
```

### All Channels Experiment

Create a script to run on all channels:

```python
from experiment import run_experiment
import pandas as pd

# Load channel list
labels_df = pd.read_csv('data/labeled_anomalies.csv')
channels = labels_df['chan_id'].tolist()

results = {}
for channel in channels:
    print(f"\n{'='*80}")
    print(f"Processing channel: {channel}")
    print(f"{'='*80}")
    
    try:
        lstm_metrics, qlstm_metrics = run_experiment(
            channel=channel,
            sequence_length=50,
            epochs=30
        )
        results[channel] = {
            'lstm': lstm_metrics,
            'qlstm': qlstm_metrics
        }
    except Exception as e:
        print(f"Error on channel {channel}: {e}")
        results[channel] = {'error': str(e)}

# Save aggregate results
import json
with open('results/all_channels_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Expected Results on Real Data

Based on published research:

| Model | F1-Score (Expected) | Note |
|-------|-------------------|------|
| Telemanom LSTM | 0.60-0.70 | Original 2018 paper |
| OmniAnomaly | 0.74 | KDD 2019 |
| USAD | 0.76 | KDD 2020 |
| GDN | 0.79 | AAAI 2021 |
| TranAD | 0.84 | VLDB 2022 (SOTA) |
| **Our LSTM** | 0.60-0.75 | Expected range |
| **Our QLSTM** | 0.55-0.70 | Expected range |

The real data is significantly more challenging than synthetic data due to:
- Complex temporal patterns
- Multiple interacting sensors
- Subtle anomalies
- Real-world noise and artifacts
- Class imbalance (~1-5% anomalies)

## Comparing Synthetic vs Real Results

| Aspect | Synthetic Data | Real NASA Data |
|--------|---------------|----------------|
| F1-Score | 95-96% | 60-80% |
| Channels | 4 channels | 55 channels |
| Features | 25 per channel | 1-25 per channel |
| Anomalies | ~5% (clear patterns) | 1-5% (subtle) |
| Training Time | ~15 min (15 epochs) | ~30 min (30 epochs) |
| Use Case | Proof of concept | Production evaluation |

## Troubleshooting

### "Kaggle credentials not found"
- Make sure `kaggle.json` is in `~/.kaggle/`
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`

### "Dataset not found"
- Verify dataset name: `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`
- Check Kaggle account has accepted dataset terms

### "Out of memory"
- Reduce batch size in experiment.py
- Process channels one at a time
- Use GPU if available

### "Channel not found"
- Check channel name matches `labeled_anomalies.csv`
- Common channels: P-1, S-1, E-1, M-1, etc.

## Next Steps

1. **Download the real data** using one of the methods above
2. **Run quick test** to verify setup: `python quick_test.py`
3. **Run full experiment** on selected channels
4. **Compare results** with published benchmarks
5. **Report findings** in RESULTS.md

## References

- Original Dataset: [NASA SMAP-MSL](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)
- Telemanom Paper: [Hundman et al., KDD 2018](https://arxiv.org/abs/1802.04431)
- GitHub Repository: [khundman/telemanom](https://github.com/khundman/telemanom)

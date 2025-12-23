# Running Experiments on Real NASA Data - Setup Guide

## Quick Setup for GitHub Actions

To run the QLSTM experiments on real NASA SMAP-MSL data automatically via GitHub Actions:

### Step 1: Get Your Kaggle API Credentials

1. Go to https://www.kaggle.com/account
2. Scroll down to the "API" section
3. Click "Create New API Token"
4. This will download a file called `kaggle.json` with your credentials

The file will look like:
```json
{
  "username": "your_username",
  "key": "your_api_key_here"
}
```

### Step 2: Add Credentials to GitHub Secrets

1. Go to your GitHub repository: https://github.com/maneetchatterjee/maneet
2. Click on **Settings** (top menu)
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add two secrets:
   - Name: `KAGGLE_USERNAME`, Value: your username from kaggle.json
   - Name: `KAGGLE_KEY`, Value: your key from kaggle.json

### Step 3: Run the Workflow

1. Go to **Actions** tab in your repository
2. Select **"Run QLSTM Experiments on Real NASA Data"** from the left sidebar
3. Click **"Run workflow"** button (top right)
4. Select branch: `copilot/implement-qlstm-on-nasa-dataset`
5. Click **"Run workflow"**

The workflow will:
- Download the real NASA SMAP-MSL dataset (55 channels)
- Run experiments on P-1 channel
- Generate all metrics and plots
- Upload results as artifacts

### Step 4: View Results

After the workflow completes:
1. Go to the workflow run page
2. Scroll down to **Artifacts** section
3. Download `experiment-results` artifact
4. Extract and view:
   - `*.json` files - Performance metrics
   - `plots/*.png` files - Visualizations

---

## Alternative: Local Setup

If you prefer to run locally:

### Step 1: Install Dependencies
```bash
cd /path/to/maneet
pip install -r requirements.txt
pip install kaggle
```

### Step 2: Setup Kaggle Credentials
```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move your downloaded kaggle.json there
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Data
```bash
python download_real_data.py
```

This will:
- Verify your Kaggle credentials
- Download NASA SMAP-MSL dataset (~50MB)
- Extract to `./data/` directory
- Validate dataset structure

### Step 4: Run Experiments
```bash
# Quick test (15 epochs, ~15 minutes)
python quick_test.py

# Full experiment (30 epochs, ~30 minutes)
python experiment.py
```

### Step 5: View Results
Results are saved to:
- `results/*.json` - Metrics (precision, recall, F1, AUC)
- `results/plots/*.png` - Visualizations

---

## Expected Results on Real Data

The real NASA SMAP-MSL dataset is significantly more challenging than synthetic data:

| Metric | Synthetic Data | Real NASA Data (Expected) |
|--------|---------------|---------------------------|
| F1-Score | 95-96% | **60-80%** |
| Precision | 99-100% | **70-85%** |
| Recall | 92-93% | **50-75%** |
| AUC | 99.99% | **80-95%** |

### Published Benchmarks (Real Data)
- Telemanom (KDD'18): F1 = 0.60-0.70
- OmniAnomaly (KDD'19): F1 = 0.74
- USAD (KDD'20): F1 = 0.76
- GDN (AAAI'21): F1 = 0.79
- **TranAD (VLDB'22): F1 = 0.84** ← State-of-the-art

Your LSTM/QLSTM should achieve F1 scores in the **0.60-0.80 range**, which would be competitive with published methods.

---

## Troubleshooting

### "Kaggle credentials not found"
- **Local**: Make sure `~/.kaggle/kaggle.json` exists with correct permissions (600)
- **GitHub Actions**: Verify secrets are added (Settings → Secrets → Actions)

### "401 Unauthorized" or "403 Forbidden"
- Regenerate your Kaggle API token
- Make sure you've accepted the dataset's terms on Kaggle website
- Visit: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

### "Dataset not found after download"
- Check `data/train/` and `data/test/` directories exist
- Verify `.npy` files are present
- Re-run `python download_real_data.py`

### "Out of memory"
- Reduce batch size in `experiment.py` (line ~128: change 128 to 64)
- Process one channel at a time
- Use `quick_test.py` with fewer epochs

---

## Dataset Information

**NASA SMAP-MSL Dataset:**
- **Total Channels:** 55
- **SMAP Channels:** 25 (Soil Moisture Active Passive satellite)
- **MSL Channels:** 27 (Mars Science Laboratory rover)
- **Features per Channel:** 1-25 (varies)
- **Anomaly Rate:** 1-5% of data points
- **Use Cases:** Spacecraft health monitoring, anomaly detection

**Data Structure:**
```
data/
├── train/
│   ├── P-1.npy ... P-25.npy  (SMAP channels)
│   ├── S-1.npy ... S-27.npy  (MSL channels)
│   └── E-1.npy, M-1.npy, etc.
├── test/
│   └── (same structure as train/)
└── labeled_anomalies.csv
```

---

## Running on All Channels

To run experiments on all 55 channels:

```python
import pandas as pd
from experiment import run_experiment

# Load channel list
labels_df = pd.read_csv('data/labeled_anomalies.csv')
channels = labels_df['chan_id'].tolist()

print(f"Found {len(channels)} channels")

results = {}
for i, channel in enumerate(channels):
    print(f"\n[{i+1}/{len(channels)}] Processing: {channel}")
    try:
        lstm_metrics, qlstm_metrics = run_experiment(
            channel=channel,
            sequence_length=50,
            epochs=30
        )
        results[channel] = {
            'lstm_f1': lstm_metrics['f1_score'],
            'qlstm_f1': qlstm_metrics['f1_score']
        }
        print(f"  LSTM F1: {lstm_metrics['f1_score']:.4f}")
        print(f"  QLSTM F1: {qlstm_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        results[channel] = {'error': str(e)}

# Save aggregate results
import json
with open('results/all_channels_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Complete! Results saved to results/all_channels_summary.json")
```

This will take several hours to complete (55 channels × 30 epochs × ~30 min = ~27.5 hours).

---

## Questions?

- See `REAL_DATA_GUIDE.md` for more details
- Check `README.md` for general usage
- Review `RESULTS.md` for expected performance analysis

**Dataset:** https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

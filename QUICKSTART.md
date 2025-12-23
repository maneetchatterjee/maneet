# Quick Start: Running on Real NASA Data

## 🎯 Goal
Run LSTM vs QLSTM experiments on the actual NASA SMAP-MSL dataset (55 channels) and get real metrics.

## ⚡ 3-Step Setup (Takes 5 minutes)

### Step 1: Get Kaggle API Credentials
```
1. Visit: https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save the downloaded kaggle.json file
```

Your kaggle.json will look like:
```json
{
  "username": "your_username_here",
  "key": "abc123def456..."
}
```

### Step 2: Add Secrets to GitHub
```
1. Go to: https://github.com/maneetchatterjee/maneet/settings/secrets/actions
2. Click "New repository secret"
3. Add:
   Name: KAGGLE_USERNAME
   Value: [copy from kaggle.json]
   
4. Click "New repository secret" again
5. Add:
   Name: KAGGLE_KEY
   Value: [copy from kaggle.json]
```

### Step 3: Run the Workflow
```
1. Go to: https://github.com/maneetchatterjee/maneet/actions
2. Click "Run QLSTM Experiments on Real NASA Data"
3. Click green "Run workflow" button
4. Select branch: copilot/implement-qlstm-on-nasa-dataset
5. Click "Run workflow"
```

## 📊 What You'll Get

After ~20-30 minutes, you'll get:

**Metrics** (downloadable artifacts):
- `lstm_P-1_metrics.json` - LSTM performance
- `qlstm_P-1_metrics.json` - QLSTM performance

**Example expected results:**
```json
{
  "precision": 0.75,
  "recall": 0.68,
  "f1_score": 0.71,    ← Compare to synthetic: 0.96
  "auc": 0.92
}
```

**Visualizations** (PNG files):
- Training curves
- Confusion matrices
- Anomaly detection plots
- Side-by-side comparison

## 🔍 Understanding Results

### Synthetic vs Real Data Performance

| Dataset | F1-Score | Why? |
|---------|----------|------|
| **Synthetic** | 96% | Simple patterns, clear anomalies |
| **Real NASA** | 60-80% | Complex, noisy, subtle anomalies |

### Comparison to Published Research

Your results should be:
- ✅ **60-70% F1** = Good (matches Telemanom KDD'18)
- ✅ **70-80% F1** = Very Good (beats OmniAnomaly, USAD, GDN)
- ✅ **>80% F1** = Excellent (approaches TranAD VLDB'22 SOTA)

## 🚨 Troubleshooting

**"Workflow not visible in Actions"**
→ Make sure you're on the correct branch: `copilot/implement-qlstm-on-nasa-dataset`

**"401 Unauthorized" error**
→ Check secrets are added correctly (no extra spaces)
→ Regenerate Kaggle API token and update secrets

**"Dataset terms not accepted"**
→ Visit https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
→ Click "Download" to accept terms

**Want to run locally instead?**
```bash
# One-time setup
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Run experiments
python download_real_data.py
python quick_test.py
```

## 📁 Files You Created

- ✅ `.github/workflows/run-real-data-experiments.yml` - Automation
- ✅ `SETUP_REAL_DATA.md` - Detailed guide (you are here)
- ✅ `download_real_data.py` - Smart downloader
- ✅ `REAL_DATA_GUIDE.md` - Comprehensive documentation

## 🎓 Next Steps

After first run on P-1 channel:
1. **Analyze results** - Compare LSTM vs QLSTM performance
2. **Run all channels** - See performance across all 55 channels
3. **Compare to papers** - Your F1 vs Telemanom (60-70%), TranAD (84%)
4. **Document findings** - Update RESULTS.md with real data metrics

---

**Need help?** Check `REAL_DATA_GUIDE.md` for more details or open an issue.

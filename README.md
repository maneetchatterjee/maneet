# Quantum LSTM for NASA Anomaly Detection

[![Run QLSTM Experiments](https://github.com/maneetchatterjee/maneet/actions/workflows/run-real-data-experiments.yml/badge.svg?branch=copilot/implement-qlstm-on-nasa-dataset)](https://github.com/maneetchatterjee/maneet/actions/workflows/run-real-data-experiments.yml)

This project implements and compares a **Quantum-Inspired LSTM (QLSTM)** with a baseline LSTM on the NASA SMAP-MSL anomaly detection dataset.

## 🚨 Want to Run on Real NASA Data?

### 🚀 **[Click Here to Run Workflow](../../actions/workflows/run-real-data-experiments.yml)** ← Direct link!

**Quick Setup:**
1. Add Kaggle credentials to [Repository Secrets](../../settings/secrets/actions)
2. Click the link above → "Run workflow" button → Select branch → Run
3. Download results after ~20-30 minutes

**Detailed guides:**
- [**QUICKSTART.md**](QUICKSTART.md) - 5-minute setup guide
- [**WORKFLOW_TROUBLESHOOTING.md**](WORKFLOW_TROUBLESHOOTING.md) - Can't see "Run workflow" button? ⭐NEW

## 🎯 Quick Results (Synthetic Data)

| Metric | Baseline LSTM | QLSTM | Winner |
|--------|---------------|-------|--------|
| **F1-Score** | **96.42%** | 95.77% | LSTM (+0.65%) |
| **Precision** | **100%** | 99.32% | LSTM (+0.68%) |
| **Recall** | **93.08%** | 92.45% | LSTM (+0.63%) |
| **AUC** | 99.99% | 99.99% | Tie |

**Key Finding:** Both models achieve excellent performance (>95% F1), with classical LSTM slightly outperforming QLSTM by <1%. No quantum advantage demonstrated at current scale (4 qubits).

⚠️ **Note:** These results are on synthetic data. For real NASA SMAP-MSL data (55 channels), expected F1: **0.60-0.80** (real data is significantly more challenging).

## 📋 Overview

The NASA SMAP-MSL dataset contains spacecraft telemetry data with labeled anomalies. This project:
- ✅ Implements a baseline LSTM model (based on Telemanom by Hundman et al., 2018)
- ✅ Implements a novel Quantum-Inspired LSTM using quantum-inspired circuits
- ✅ Compares performance on anomaly detection tasks
- ✅ Provides comprehensive metrics and visualizations
- ✅ Includes literature review and comparison with state-of-the-art
- ✅ **GitHub Actions workflow for automated experiments on real data** ⭐NEW

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet

# Install dependencies
pip install -r requirements.txt
```

### Option 1: Synthetic Data (Quick Demo)

```bash
# Generate synthetic NASA-like data
python generate_synthetic_data.py

# Run full experiment (30 epochs, ~30 minutes)
python experiment.py

# Or run quick test (15 epochs, ~15 minutes)
python quick_test.py
```

### Option 2: Real NASA SMAP-MSL Data (Production)

For running on the actual NASA dataset (55 channels, real spacecraft telemetry):

```bash
# 1. Set up Kaggle API credentials (one-time setup)
#    See REAL_DATA_GUIDE.md for detailed instructions

# 2. Download real NASA data
python download_real_data.py

# 3. Run experiments on real data
python experiment.py    # Will automatically use real data if available
```

**📖 See [REAL_DATA_GUIDE.md](REAL_DATA_GUIDE.md) for:**
- Kaggle setup instructions
- Alternative download methods
- Expected results on real data (F1: 0.60-0.80)
- Running on all 55 channels
- Troubleshooting guide

Results will be saved to:
- `results/*.json` - Performance metrics
- `results/plots/*.png` - Visualizations

## 📊 Results Summary

### Performance Metrics (Synthetic Data)

**Baseline LSTM:**
- F1-Score: 96.42% | Precision: 100% | Recall: 93.08% | AUC: 99.99%
- Confusion Matrix: 2791 TN, 0 FP, 11 FN, 148 TP

**Quantum-Inspired LSTM:**
- F1-Score: 95.77% | Precision: 99.32% | Recall: 92.45% | AUC: 99.99%
- Confusion Matrix: 2790 TN, 1 FP, 12 FN, 147 TP

### Comparison with Literature

| Model | Year | Venue | F1-Score | Data Type |
|-------|------|-------|----------|-----------|
| Telemanom (LSTM) | 2018 | KDD | 0.60-0.70 | Real NASA |
| OmniAnomaly | 2019 | KDD | 0.74 | Real NASA |
| USAD | 2020 | KDD | 0.76 | Real NASA |
| GDN | 2021 | AAAI | 0.79 | Real NASA |
| TranAD | 2022 | VLDB | 0.84 | Real NASA |
| **Our LSTM** | 2024 | - | **0.9642** | **Synthetic** |
| **Our QLSTM** | 2024 | - | **0.9577** | **Synthetic** |

⚠️ **Note:** Higher scores due to synthetic data; not directly comparable. For real NASA data experiments, download the dataset using `download_real_data.py` - expected F1: 0.60-0.80.

## 🔬 Technical Details

### Dataset Options

**1. Synthetic Data (Current Results):**
- **Source:** Generated NASA-like spacecraft telemetry
- **Training:** 7,950 sequences × 25 features
- **Testing:** 2,950 sequences × 25 features
- **Anomalies:** ~5% of test data (clear patterns)
- **Generation:** Trend + seasonality + noise + injected anomalies
- **Performance:** F1 > 95% (easier than real data)

**2. Real NASA SMAP-MSL Data (Recommended for Production):**
- **Source:** Actual spacecraft telemetry from SMAP & MSL missions
- **Channels:** 55 channels (25 SMAP + 27 MSL + 3 others)
- **Features:** 1-25 per channel (varies)
- **Anomalies:** 1-5% (subtle, realistic)
- **Download:** `python download_real_data.py` (requires Kaggle API)
- **Performance:** Expected F1: 0.60-0.80 (matches published research)
- **Guide:** See [REAL_DATA_GUIDE.md](REAL_DATA_GUIDE.md)

### Baseline LSTM Architecture
```
Input (50, 25) → LSTM(80) → Dropout(0.2) → LSTM(80) → Dropout(0.2) → Dense(25)
```
- Parameters: ~52K
- Batch size: 128
- Optimizer: Adam (lr=0.001)

### QLSTM Architecture
```
Input (50, 25) → LSTM(80) → Dropout(0.2) → QuantumLSTM(80) → Dropout(0.2) → Dense(25)
```
- Parameters: ~54K (includes quantum-inspired parameters)
- Quantum component: 4 "qubits", 2 layers
- Operations: Parameterized rotations + entanglement-like mixing
- Integration: 70% classical + 30% quantum-inspired
- Batch size: 64
- Optimizer: Adam (lr=0.001)

## 📁 Project Structure

```
maneet/
├── README.md                      # This file - Quick start guide
├── RESULTS.md                     # Detailed results and analysis (15+ pages)
├── SUMMARY.md                     # Executive summary (8 pages)
├── REAL_DATA_GUIDE.md            # Real NASA data guide (6 pages) ⭐NEW
├── requirements.txt               # Python dependencies
├── data_loader.py                 # Data loading (supports Kaggle) ⭐UPDATED
├── download_real_data.py          # Real data downloader ⭐NEW
├── generate_synthetic_data.py     # Synthetic data generation
├── baseline_lstm.py               # Classical LSTM implementation
├── qlstm.py                      # Quantum-inspired LSTM implementation
├── experiment.py                  # Main experiment script
├── quick_test.py                  # Quick test (reduced epochs)
└── results/                       # Generated results (gitignored)
    ├── *.json                     # Metrics files
    └── plots/                     # Visualizations
        ├── lstm_*_training.png    # LSTM training curves
        ├── lstm_*_results.png     # LSTM anomaly detection
        ├── lstm_*_confusion.png   # LSTM confusion matrix
        ├── qlstm_*_training.png   # QLSTM training curves
        ├── qlstm_*_results.png    # QLSTM anomaly detection
        ├── qlstm_*_confusion.png  # QLSTM confusion matrix
        └── comparison_*.png       # Side-by-side comparison
```

## 📚 Documentation

- **[README.md](README.md)** - This file - Quick start and overview
- **[REAL_DATA_GUIDE.md](REAL_DATA_GUIDE.md)** - **⭐NEW** Complete guide for real NASA data (6 pages)
  - Three download methods (automated/manual/CLI)
  - Kaggle setup instructions
  - Expected results on real data
  - Troubleshooting guide
  - All 55 channels processing
- **[SUMMARY.md](SUMMARY.md)** - Executive summary with key findings (8 pages)
- **[RESULTS.md](RESULTS.md)** - Comprehensive analysis with literature review (15+ pages)
  - Dataset overview and characteristics
  - Literature review (6 papers, 2018-2022)
  - Experimental setup and methodology
  - Detailed results and metrics
  - Analysis and interpretation
  - Comparison with state-of-the-art
  - Limitations and future work
  - Complete conclusions

## 🔑 Key Findings

### What Works (Synthetic Data)
✅ **Both models achieve excellent performance** (>95% F1-score)
✅ **Near-perfect anomaly discrimination** (>99.9% AUC)
✅ **Very low false positive rates** (0-1 false alarms)
✅ **Stable training** (converges in ~10 epochs)
✅ **Quantum-inspired features integrate successfully** (no training instability)

### What We Learned
⚠️  **Classical LSTM slightly outperforms QLSTM** (<1% difference on synthetic data)
⚠️  **No quantum advantage at small scale** (4 "qubits")
⚠️  **True quantum circuits difficult to integrate** (TensorFlow constraints)
⚠️  **Synthetic data easier than real NASA data** (F1: 0.96 vs expected 0.60-0.80)
⚠️  **Real data validation needed** (synthetic results not comparable to published work)

### Implications
📌 **For Production:** Test on real NASA data first - use `download_real_data.py`
📌 **For Research:** QLSTM shows promise but needs larger quantum circuits (16+ qubits)
📌 **For Quantum ML:** Hybrid approaches practical, but quantum advantage unproven at this scale
📌 **Current Status:** Proof of concept on synthetic data; real data tooling ready

## 🔮 Future Work

### Immediate
- [ ] **Test on real NASA SMAP-MSL dataset** (all 55 channels) - **Tools ready!**
- [ ] Extend training to 30-50 epochs
- [ ] Ablation studies on quantum component size

### Advanced
- [ ] Scale up to 8-16 qubit quantum circuits
- [ ] Deploy on real quantum hardware (IBM Quantum, AWS Braket)
- [ ] Implement fully quantum LSTM
- [ ] Explore quantum attention mechanisms
- [ ] Benchmark against TranAD and GDN

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.8+ (for deep learning)
- PennyLane 0.28+ (for quantum circuits)
- NumPy, Pandas, Scikit-learn (for data processing)
- Matplotlib, Seaborn (for visualization)

See `requirements.txt` for complete list.

## 📖 Citation

If you use this code, please cite the original Telemanom paper:

```bibtex
@inproceedings{hundman2018telemanom,
  title={Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding},
  author={Hundman, Kyle and Constantinou, Valentino and Laporte, Christopher and Colwell, Ian and Soderstrom, Tom},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={387--395},
  year={2018}
}
```

## 🙏 Acknowledgments

- NASA for the SMAP-MSL dataset
- Kyle Hundman et al. for the original Telemanom work
- PennyLane team for quantum computing framework
- TensorFlow team for deep learning framework

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 📄 License

MIT License - See LICENSE file for details

---

**Status:** ✅ Complete
**Last Updated:** December 2024
**Experiment Time:** ~15 minutes (quick test) / ~30 minutes (full)
**Hardware:** CPU (quantum simulation)
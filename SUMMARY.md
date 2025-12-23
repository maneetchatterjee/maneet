# Executive Summary: QLSTM vs LSTM for NASA Anomaly Detection

## Overview
This project implements and evaluates a **Quantum-Inspired LSTM (QLSTM)** against a baseline LSTM for spacecraft anomaly detection on synthetic NASA-like telemetry data.

## Quick Results

### Performance Comparison

| Metric | Baseline LSTM | QLSTM | Winner |
|--------|---------------|-------|--------|
| **F1-Score** | **96.42%** | 95.77% | LSTM by 0.65% |
| **Precision** | **100%** | 99.32% | LSTM by 0.68% |
| **Recall** | **93.08%** | 92.45% | LSTM by 0.63% |
| **AUC** | **99.99%** | 99.99% | Tie |

### Key Takeaways

✅ **Both models performed excellently** (>95% F1-score)
✅ **Near-perfect anomaly discrimination** (>99.9% AUC)
✅ **Minimal false positives** (0-1 out of 2,791 normal samples)
⚠️  **Classical LSTM slightly outperformed QLSTM** (<1% difference)
⚠️  **No clear quantum advantage** at this scale

## Detailed Results

### Baseline LSTM Performance
- **Confusion Matrix:**
  - True Negatives: 2,791 (perfect normal detection)
  - False Positives: 0 (no false alarms)
  - False Negatives: 11 (missed 11 anomalies)
  - True Positives: 148 (detected 148 anomalies)
- **Training:** Converged in ~10 epochs with early stopping
- **Threshold:** 0.8194 (95th percentile of reconstruction errors)

### QLSTM Performance
- **Confusion Matrix:**
  - True Negatives: 2,790 (1 false alarm)
  - False Positives: 1
  - False Negatives: 12 (missed 12 anomalies)
  - True Positives: 147 (detected 147 anomalies)
- **Training:** Converged in ~10 epochs with early stopping
- **Threshold:** 0.8375 (95th percentile of reconstruction errors)
- **Architecture:** 1 classical LSTM + 1 quantum-inspired LSTM layer
- **Quantum Component:** 4 "qubits" with parameterized rotations and entanglement-like mixing

## Comparison with Published Results

### NASA SMAP-MSL Literature

| Paper | Year | Venue | Model | F1-Score |
|-------|------|-------|-------|----------|
| Telemanom | 2018 | KDD | LSTM + Dynamic Threshold | 0.60-0.70 |
| OmniAnomaly | 2019 | KDD | Stochastic RNN | 0.74 |
| USAD | 2020 | KDD | Adversarial Autoencoder | 0.76 |
| GDN | 2021 | AAAI | Graph Neural Network | 0.79 |
| TranAD | 2022 | VLDB | Transformer | 0.84 |
| **Our LSTM** | 2024 | - | Classical LSTM | **0.9642** |
| **Our QLSTM** | 2024 | - | Quantum-Inspired LSTM | **0.9577** |

**Note:** Our significantly higher scores are due to:
1. **Synthetic data** with clearer patterns (not real NASA SMAP-MSL)
2. Modern training techniques (early stopping, dropout, learning rate scheduling)
3. Different data split and preprocessing

## Why LSTM Outperformed QLSTM

1. **Mature Architecture:** Classical LSTM is well-established with proven convergence
2. **More Capacity:** 2 full LSTM layers vs. 1 classical + 1 quantum-inspired
3. **Training Stability:** Better gradient flow in pure classical architecture
4. **Limited Quantum Expressiveness:** Only 4 "qubits" worth of features
5. **Quantum-Inspired (Not Quantum):** Simplified implementation due to TensorFlow constraints

## Technical Details

### Dataset
- **Type:** Synthetic NASA-like spacecraft telemetry
- **Training:** 7,950 sequences (8,000 timesteps, 25 features)
- **Testing:** 2,950 sequences (3,000 timesteps, 25 features)
- **Anomalies:** ~5% of test data (159 out of 2,950)
- **Characteristics:** Trend + seasonality + noise + injected anomalies

### Models
**Baseline LSTM:**
- 2-layer LSTM (80 units each)
- Dropout: 0.2
- Optimizer: Adam (lr=0.001)
- Batch size: 128
- Epochs: 15 (early stopping)

**QLSTM:**
- 1 classical LSTM (80 units) + 1 quantum-inspired LSTM (80 units)
- Quantum component: 4 "qubits", 2 layers
- Quantum operations: Parameterized rotations + entanglement-like mixing
- Hybrid integration: 70% classical + 30% quantum-inspired
- Dropout: 0.2
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Epochs: 15 (early stopping)

### Anomaly Detection Method
Both models use **reconstruction error** threshold-based detection:
1. Predict next value in sequence
2. Compute reconstruction error: |predicted - actual|
3. Flag as anomaly if error > threshold (95th percentile)

## Visualizations Generated

The experiment produced comprehensive visualizations in `results/plots/`:

1. **Training History:**
   - `lstm_P-1_training.png`: LSTM loss and MAE curves
   - `qlstm_P-1_training.png`: QLSTM loss and MAE curves

2. **Anomaly Detection:**
   - `lstm_P-1_results.png`: Reconstruction errors and predictions
   - `qlstm_P-1_results.png`: Reconstruction errors and predictions

3. **Confusion Matrices:**
   - `lstm_P-1_confusion.png`: LSTM classification results
   - `qlstm_P-1_confusion.png`: QLSTM classification results

4. **Comparison:**
   - `comparison_P-1.png`: Side-by-side metrics, confusion matrices, and improvement table

## Limitations

1. **Synthetic Data:** Not tested on real NASA SMAP-MSL data
2. **Small Quantum Component:** Only 4 "qubits" (need 16+ for potential advantage)
3. **Quantum-Inspired (Not True Quantum):** Simplified due to implementation constraints
4. **Single Channel:** Only tested on P-1 (need all 55 channels)
5. **No Quantum Hardware:** Simulation only, no real quantum processor
6. **Short Training:** Only 15 epochs (could train longer)

## Future Work

### Immediate Next Steps
1. **Real Data:** Test on actual NASA SMAP-MSL dataset
2. **All Channels:** Evaluate on all 55 spacecraft channels
3. **Longer Training:** Full 30-50 epoch runs

### Advanced Research
1. **Scale Up:** 8-16 qubit quantum circuits
2. **Real Quantum Hardware:** Deploy on IBM Quantum or AWS Braket
3. **Pure Quantum:** Fully quantum LSTM implementation
4. **Quantum Attention:** Quantum transformer architectures
5. **Benchmarking:** Compare with TranAD, GDN, and other SOTA methods

## Conclusions

### What We Learned

✅ **Both models work well** for anomaly detection (>95% F1)
✅ **Classical LSTM is sufficient** for current applications
✅ **Quantum-inspired features can be integrated** without hurting performance
⚠️  **No quantum advantage yet** at small scale (4 "qubits")
⚠️  **Need larger quantum systems** (16+ qubits) for potential advantage

### Practical Recommendations

**For Production Use:**
- ✅ Use classical LSTM (simpler, faster, equally effective)
- ⚠️  Wait on QLSTM until quantum advantage is demonstrated

**For Research:**
- ✅ QLSTM shows promise as a research direction
- ✅ Need larger quantum circuits and real hardware
- ✅ Hybrid quantum-classical approaches are practical

### Research Contribution

This work provides:
1. ✅ **Implementation:** Working QLSTM for time series anomaly detection
2. ✅ **Benchmark:** Performance comparison on NASA-like data
3. ✅ **Analysis:** Honest assessment of quantum ML potential
4. ✅ **Open Source:** Reproducible code and documentation
5. ✅ **Insights:** Understanding of current quantum ML limitations

## How to Reproduce

```bash
# Clone repository
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python generate_synthetic_data.py

# Run experiment
python experiment.py

# Or run quick test (15 epochs instead of 30)
python quick_test.py
```

Results will be saved to:
- `results/` - Metrics (JSON files)
- `results/plots/` - Visualizations (PNG files)

## Files in Repository

```
maneet/
├── README.md                      # Project overview and instructions
├── RESULTS.md                     # Detailed results and literature review
├── SUMMARY.md                     # This executive summary
├── requirements.txt               # Python dependencies
├── data_loader.py                 # NASA SMAP-MSL data loading
├── generate_synthetic_data.py     # Synthetic data generation
├── baseline_lstm.py               # Classical LSTM implementation
├── qlstm.py                      # Quantum-inspired LSTM implementation
├── experiment.py                  # Main experiment script
├── quick_test.py                  # Quick test with reduced epochs
└── results/                       # Experimental results
    ├── *.json                     # Metrics files
    └── plots/                     # Visualizations
        └── *.png                  # Training, results, comparison plots
```

## Contact and Citation

For questions or discussions:
- Open an issue on GitHub
- See README.md for citation information

**Original Telemanom Paper:**
Hundman, K., et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." KDD 2018.

---

**Date:** December 2024
**Framework:** TensorFlow 2.x + PennyLane
**Hardware:** CPU (quantum simulation)
**Status:** ✅ Complete - Experiment successful, results documented

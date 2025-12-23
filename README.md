# Quantum LSTM for NASA Anomaly Detection

This project implements and compares a Quantum LSTM (QLSTM) with a baseline LSTM on the NASA SMAP-MSL anomaly detection dataset.

## Overview

The NASA SMAP-MSL dataset contains spacecraft telemetry data with labeled anomalies. This project:
- Implements a baseline LSTM model (based on Telemanom by Hundman et al., 2018)
- Implements a novel Quantum LSTM using quantum circuits
- Compares performance on anomaly detection tasks
- Provides comprehensive metrics and visualizations

## Features

- **Data Loading:** Automated download and preprocessing of NASA SMAP-MSL dataset
- **Baseline LSTM:** Classical 2-layer LSTM for comparison
- **Quantum LSTM:** Hybrid quantum-classical LSTM with PennyLane
- **Comprehensive Evaluation:** Precision, Recall, F1-Score, AUC metrics
- **Visualizations:** Training curves, confusion matrices, comparison plots
- **Literature Review:** Summary of published results on this dataset

## Installation

```bash
# Clone the repository
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete experiment:

```bash
python experiment.py
```

This will:
1. Download the NASA SMAP-MSL dataset
2. Train both LSTM and QLSTM models
3. Evaluate performance
4. Generate comparison plots
5. Save results to `results/` directory

## Project Structure

```
maneet/
├── data_loader.py          # NASA SMAP-MSL data loading
├── baseline_lstm.py        # Classical LSTM implementation
├── qlstm.py               # Quantum LSTM implementation
├── experiment.py          # Main experiment script
├── requirements.txt       # Python dependencies
├── RESULTS.md            # Detailed results and literature review
└── results/              # Experimental results (generated)
    ├── plots/            # Visualizations
    └── *.json            # Metrics files
```

## Dataset

The NASA SMAP-MSL dataset includes:
- **SMAP:** Soil Moisture Active Passive satellite data (25 channels)
- **MSL:** Mars Science Laboratory rover data (28 MSL + 2 Curiosity channels)
- **Total:** 55 channels of multivariate time series
- **Labels:** Expert-labeled anomalies

Source: https://github.com/khundman/telemanom

## Models

### Baseline LSTM
- 2-layer LSTM with 80 units each
- Dropout (0.2)
- Reconstruction-based anomaly detection
- Based on KDD 2018 Telemanom paper

### Quantum LSTM (QLSTM)
- Hybrid quantum-classical architecture
- 1 classical LSTM + 1 quantum LSTM layer
- 4 qubits, 2-layer quantum circuit
- PennyLane quantum simulation
- Novel approach combining quantum computing with time series analysis

## Results

Detailed results, metrics, and comparison with literature can be found in [RESULTS.md](RESULTS.md).

Key comparisons:
- Performance metrics (F1, Precision, Recall, AUC)
- Training efficiency
- Anomaly detection capability
- Comparison with state-of-the-art methods

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- PennyLane 0.28+ (for quantum circuits)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

See `requirements.txt` for complete list.

## Citation

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

## License

MIT License - See LICENSE file for details

## Acknowledgments

- NASA for the SMAP-MSL dataset
- Kyle Hundman et al. for the original Telemanom work
- PennyLane team for quantum computing framework

## Contact

For questions or issues, please open an issue on GitHub.
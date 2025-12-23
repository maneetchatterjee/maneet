# QLSTM on NASA SMAP-MSL Anomaly Detection Dataset

## Literature Review and Baseline Comparison

### Dataset Overview
The NASA SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory) dataset is a benchmark for anomaly detection in spacecraft telemetry data. It contains real spacecraft sensor data with labeled anomalies.

**Dataset Characteristics:**
- **Source:** NASA spacecraft telemetry
- **Channels:** 55 channels (25 SMAP + 28 MSL)
- **Type:** Multivariate time series
- **Task:** Unsupervised anomaly detection
- **Challenge:** Highly imbalanced (anomalies < 5%)

### Published Results on SMAP-MSL Dataset

#### 1. **Telemanom (Hundman et al., 2018)** - Original Work
- **Paper:** "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
- **Venue:** KDD 2018
- **Model:** LSTM with dynamic thresholding
- **Results:**
  - F1-Score: ~0.60-0.70 (varies by channel)
  - Precision: ~0.70-0.80
  - Recall: ~0.50-0.60
- **Architecture:** 
  - 2-layer LSTM (80 units each)
  - Dropout (0.3)
  - Prediction horizon: single step
- **Key Innovation:** Nonparametric dynamic thresholding using EWMA and pruning
- **GitHub:** khundman/telemanom

#### 2. **OmniAnomaly (Su et al., 2019)**
- **Paper:** "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"
- **Venue:** KDD 2019
- **Model:** Stochastic RNN with variational inference
- **Results:**
  - F1-Score: ~0.74 (average)
  - Improved over Telemanom
- **Key Innovation:** Planar normalizing flow and stochastic variable connection

#### 3. **USAD (Audibert et al., 2020)**
- **Paper:** "USAD: UnSupervised Anomaly Detection on Multivariate Time Series"
- **Venue:** KDD 2020
- **Model:** Adversarial autoencoder
- **Results:**
  - F1-Score: ~0.76-0.78
  - Faster training than RNN-based methods
- **Key Innovation:** Adversarially trained autoencoder with isolation-based loss

#### 4. **GDN (Deng & Hooi, 2021)**
- **Paper:** "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series"
- **Venue:** AAAI 2021
- **Model:** Graph Deviation Network
- **Results:**
  - F1-Score: ~0.77-0.80
  - Better capture of inter-sensor relationships
- **Key Innovation:** Graph structure learning for sensor relationships

#### 5. **TranAD (Tuli et al., 2022)**
- **Paper:** "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series"
- **Venue:** VLDB 2022
- **Model:** Transformer-based
- **Results:**
  - F1-Score: ~0.82-0.84
  - State-of-the-art on SMAP-MSL
- **Key Innovation:** Self-attention mechanism with adversarial training

#### 6. **Recent Deep Learning Approaches (2023)**
- Various transformer and attention-based models
- F1-Scores ranging from 0.80-0.85
- Focus on interpretability and efficiency

### Summary of State-of-the-Art
| Model | Year | F1-Score | Precision | Recall | Key Feature |
|-------|------|----------|-----------|--------|-------------|
| Telemanom (LSTM) | 2018 | 0.60-0.70 | 0.70-0.80 | 0.50-0.60 | Baseline LSTM |
| OmniAnomaly | 2019 | 0.74 | 0.78 | 0.70 | Stochastic RNN |
| USAD | 2020 | 0.76 | 0.80 | 0.73 | Adversarial AE |
| GDN | 2021 | 0.79 | 0.82 | 0.76 | Graph structure |
| TranAD | 2022 | 0.84 | 0.87 | 0.81 | Transformer |

---

## Quantum LSTM (QLSTM) Approach

### Motivation
Quantum computing offers potential advantages for machine learning:
1. **Quantum superposition:** Explore multiple states simultaneously
2. **Quantum entanglement:** Capture complex correlations
3. **Quantum interference:** Amplify relevant patterns
4. **Potential quantum speedup:** For certain operations

### QLSTM Architecture

#### Design
Our QLSTM implementation combines classical LSTM with quantum circuits:

```
Input → Classical LSTM → Quantum LSTM Cell → Output
         (80 units)      (4 qubits, 2 layers)
```

**Quantum Circuit Structure:**
1. **Encoding Layer:** Map classical data to quantum states using RY rotations
2. **Parametrized Quantum Circuit (PQC):**
   - Rotation gates (RX, RY, RZ) on each qubit
   - CNOT gates for entanglement
   - 2 layers of quantum operations
3. **Measurement:** Pauli-Z expectation values
4. **Integration:** Weighted combination with classical hidden state (70% classical, 30% quantum)

**Parameters:**
- 4 qubits (limited by simulation complexity)
- 2 quantum layers
- 80 LSTM units
- Batch size: 64 (vs 128 for classical LSTM)

#### Limitations of Current Implementation
1. **Small quantum circuit:** Only 4 qubits due to simulation constraints
2. **Batch processing:** Limited quantum batch processing for efficiency
3. **Hybrid approach:** Heavily weighted toward classical computation
4. **No quantum advantage:** Current implementation likely doesn't achieve quantum speedup

### Expected Behavior

Given the constraints of our implementation:
- **If QLSTM performs better:** May indicate quantum circuits help capture patterns
- **If QLSTM performs similarly:** Quantum component may not add significant value at this scale
- **If QLSTM performs worse:** Could be due to:
  - Limited quantum expressiveness (only 4 qubits)
  - Training difficulty (more parameters)
  - Quantum measurement noise

---

## Experimental Setup

### Dataset Configuration
- **Channel:** P-1 (SMAP channel)
- **Sequence Length:** 50 timesteps
- **Training/Test Split:** As provided in original dataset
- **Normalization:** Z-score normalization using training statistics

### Model Configuration

**Baseline LSTM:**
- Architecture: 2-layer LSTM (80 units each)
- Dropout: 0.2
- Optimizer: Adam (lr=0.001)
- Batch size: 128
- Epochs: 30 (with early stopping)
- Loss: MSE (mean squared error)

**QLSTM:**
- Architecture: 1 classical LSTM + 1 QLSTM (80 units)
- Quantum: 4 qubits, 2 layers
- Dropout: 0.2
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Epochs: 30 (with early stopping)
- Loss: MSE

### Evaluation Metrics
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** Harmonic mean of precision and recall
- **AUC:** Area under ROC curve
- **Threshold:** 95th percentile of reconstruction errors

### Anomaly Detection Method
Both models use reconstruction error-based anomaly detection:
1. Predict next value in sequence
2. Compute reconstruction error: |predicted - actual|
3. Flag as anomaly if error > threshold

---

## Results

*Results will be populated after running the experiment.*

### Performance Metrics

#### Baseline LSTM
- Precision: TBD
- Recall: TBD
- F1-Score: TBD
- AUC: TBD

#### QLSTM
- Precision: TBD
- Recall: TBD
- F1-Score: TBD
- AUC: TBD

### Comparison
- Improvement in F1-Score: TBD
- Improvement in Precision: TBD
- Improvement in Recall: TBD
- Improvement in AUC: TBD

---

## Analysis and Discussion

### Interpretation of Results
*To be filled after running experiments*

### Comparison with Literature
*To be filled after running experiments*

### Quantum Component Analysis
*To be filled after running experiments*

### Limitations
1. **Small-scale quantum simulation:** Only 4 qubits
2. **No real quantum hardware:** Simulation only
3. **Single channel evaluation:** Only tested on P-1
4. **Hybrid architecture:** Heavy reliance on classical components
5. **No quantum advantage demonstration:** Likely no speedup at this scale

### Future Work
1. **Scale up:** Test with more qubits (8-16) if resources allow
2. **Multiple channels:** Evaluate on all SMAP-MSL channels
3. **Pure quantum LSTM:** Explore fully quantum implementations
4. **Quantum kernel methods:** Alternative quantum ML approaches
5. **Real quantum hardware:** Test on actual quantum computers
6. **Advanced quantum circuits:** More sophisticated quantum architectures

---

## Conclusion

This experiment implements and evaluates a Quantum LSTM (QLSTM) on the NASA SMAP-MSL anomaly detection dataset, comparing it against the baseline LSTM from the original Telemanom work.

**Key Findings:**
*To be summarized after running experiments*

**Practical Implications:**
- Quantum machine learning for time series anomaly detection is an emerging field
- Current quantum simulators have limitations (4-16 qubits typically)
- Hybrid quantum-classical approaches are most practical today
- True quantum advantage requires larger quantum circuits and specialized problems

**Research Contribution:**
This work provides:
1. Implementation of QLSTM for spacecraft anomaly detection
2. Benchmark comparison on real-world dataset
3. Analysis of quantum ML applicability to time series
4. Open-source code for reproduction

---

## References

1. Hundman, K., et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." KDD 2018.

2. Su, Y., et al. (2019). "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." KDD 2019.

3. Audibert, J., et al. (2020). "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." KDD 2020.

4. Deng, A., & Hooi, B. (2021). "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series." AAAI 2021.

5. Tuli, S., et al. (2022). "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series." VLDB 2022.

6. Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature Hilbert Spaces." Physical Review Letters.

7. Cong, I., et al. (2019). "Quantum convolutional neural networks." Nature Physics.

---

## Appendix

### Code Structure
```
maneet/
├── data_loader.py          # NASA SMAP-MSL data loading
├── baseline_lstm.py        # Classical LSTM implementation
├── qlstm.py               # Quantum LSTM implementation
├── experiment.py          # Main experiment script
├── requirements.txt       # Python dependencies
└── results/              # Experimental results
    ├── plots/            # Visualizations
    └── models/           # Saved models
```

### How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment
python experiment.py
```

### System Requirements
- Python 3.8+
- TensorFlow 2.8+
- PennyLane 0.28+
- 8GB+ RAM
- GPU recommended (optional)

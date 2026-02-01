# Research-Grade Change Detection System - Complete Guide

## Overview

This repository implements a research-grade change detection system for remote sensing with out-of-distribution (OOD) awareness and epistemic uncertainty estimation, suitable for submission to CVPR EarthVision / ICLR ML4RS.

## Key Features

### Architecture
- **Siamese CNN**: Shared-weight ResNet-18/34 encoders following Daudt et al. (IGARSS 2018)
- **Change Embedding**: Latent representation z_Δ capturing how scenes change
- **Binary Classifier**: Standard BCE-based change detection
- **Density Models**: GMM and Normalizing Flow for in-distribution modeling
- **Deep Ensembles**: Epistemic uncertainty via model disagreement (UnCRtainTS-inspired)
- **Three-Way Decision**: no-change / change (confident) / change (OOD/abstain)

### Datasets
- **OSCD** (Onera Satellite Change Detection): Sentinel-2 bi-temporal imagery
- **LEVIR-CD**: High-resolution building change detection
- Region-wise splits to prevent pixel leakage

### Training
- Mixed precision (AMP) support
- Config-driven experiments (YAML)
- Checkpoint management
- Deterministic runs with seed control

### Evaluation
- **Change Detection**: Precision, Recall, F1, IoU
- **OOD Detection**: AUROC, FPR at TPR, likelihood histograms
- **Calibration**: ECE, MCE, Brier score, reliability diagrams
- **Three-Way**: Coverage, confident accuracy, abstention rate
- **Risk-Coverage**: Trade-off curves

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Quick Start

```bash
# Clone repository
git clone <repository_url>
cd maneet

# Install dependencies
pip install -r requirements.txt

# Or use the setup script
chmod +x setup.sh
./setup.sh

# Validate installation
python test_system.py
```

## Dataset Preparation

### OSCD Dataset

1. Download OSCD dataset from [official source]
2. Organize as:
```
data/OSCD/
├── train/
│   ├── images/
│   │   ├── city1_t1.tif
│   │   ├── city1_t2.tif
│   │   └── ...
│   └── labels/
│       ├── city1_label.tif
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### LEVIR-CD Dataset

1. Download LEVIR-CD dataset
2. Organize as:
```
data/LEVIR-CD/
├── train/
│   ├── A/          # Time 1 images
│   ├── B/          # Time 2 images
│   └── label/
├── val/
│   └── ...
└── test/
    └── ...
```

## Training

### Basic Training

```bash
python experiments/train.py --config experiments/configs/oscd_baseline.yaml
```

### Configuration Files

Example configuration (`oscd_baseline.yaml`):

```yaml
experiment:
  name: "oscd_baseline_gmm"
  seed: 42
  output_dir: "experiments/runs"

model:
  backbone: "resnet18"
  embedding_dim: 128
  density_model: "gmm"  # or "normalizing_flow" or null
  n_gmm_components: 3
  pretrained: true

training:
  batch_size: 16
  epochs: 100
  optimizer:
    type: "adamw"
    lr: 0.0001
  loss:
    type: "combined"  # bce, dice, combined, or focal
  mixed_precision: true
```

### Ensemble Training

For ensemble models, set in config:

```yaml
ensemble:
  enabled: true
  n_models: 5
```

### Resume Training

```bash
python experiments/train.py \
    --config experiments/configs/oscd_baseline.yaml \
    --resume experiments/runs/oscd_baseline_20240201/checkpoints/checkpoint_epoch_50.pth
```

## Evaluation

### Run Evaluation

```bash
python experiments/evaluate.py \
    --config experiments/configs/oscd_baseline.yaml \
    --checkpoint experiments/runs/oscd_baseline_20240201/checkpoints/best_model.pth \
    --output results/evaluation_results.json
```

### Evaluation Output

Results are saved in JSON format with:
- Change detection metrics (P, R, F1, IoU)
- OOD detection metrics (AUROC, AUPR, FPR)
- Calibration metrics (ECE, MCE, Brier)
- Three-way decision metrics
- Risk-coverage curves

## Model Architecture Details

### 1. Siamese Encoder

```python
from models import build_siamese_resnet

encoder = build_siamese_resnet(
    backbone='resnet18',  # or 'resnet34'
    pretrained=True,
    input_channels=3
)

# Process bi-temporal images
f1, f2 = encoder(image_t1, image_t2)
```

### 2. Change Embedding

```python
from models import ChangeDetectionHead

head = ChangeDetectionHead(
    input_dim=512,
    embedding_dim=128
)

# Compute change embedding and classification
logits, z_delta = head(feature_diff, return_embedding=True)
```

### 3. Density Models

#### GMM
```python
from models.density_models import build_change_gmm

gmm = build_change_gmm(n_components=3)
gmm.fit(train_change_embeddings)
log_liks = gmm.log_likelihood(test_embeddings)
```

#### Normalizing Flow
```python
from models.density_models import build_normalizing_flow

flow = build_normalizing_flow(input_dim=128, n_flows=6)
flow.fit(train_change_embeddings)
log_liks = flow.log_likelihood(test_embeddings)
```

### 4. Complete Model

```python
from models import build_change_detection_model

model = build_change_detection_model(
    backbone='resnet18',
    embedding_dim=128,
    density_model_type='gmm'
)

# Forward pass
logits, z_delta = model(img_t1, img_t2, return_embedding=True)

# Three-way decision
decisions, info = model.three_way_decision(img_t1, img_t2)
# decisions: 0 = no-change, 1 = change (confident), 2 = abstain
```

### 5. Deep Ensemble

```python
from models import build_ensemble, ChangeDetectionModel

ensemble = build_ensemble(
    model_class=ChangeDetectionModel,
    model_kwargs={'backbone': 'resnet18', 'embedding_dim': 128},
    n_models=3
)

# Predict with uncertainty
mean_prob, aleatoric, epistemic = ensemble.predict_with_uncertainty(img_t1, img_t2)
```

## Ablation Studies

The codebase supports the following ablations:

1. **No Density Model**: Set `density_model: null` in config
2. **GMM vs Flow**: Set `density_model: "gmm"` or `"normalizing_flow"`
3. **Single vs Ensemble**: Set `ensemble.enabled: true/false`
4. **With/Without Abstention**: Adjust thresholds or disable OOD check

## Code Structure

```
.
├── models/                     # Model architectures
│   ├── siamese_resnet.py      # Siamese encoder
│   ├── change_embedding.py    # Change embedding + classifier
│   ├── density_models/        # GMM and Flow
│   ├── ensemble.py            # Deep ensemble
│   └── change_detection_model.py  # Complete model
│
├── datasets/                   # Dataset loaders
│   ├── oscd.py                # OSCD loader
│   └── levir_cd.py            # LEVIR-CD loader
│
├── training/                   # Training infrastructure
│   ├── trainer.py             # Trainer with AMP
│   └── losses.py              # Loss functions
│
├── evaluation/                 # Metrics and evaluation
│   ├── metrics.py             # Change detection metrics
│   ├── ood_metrics.py         # OOD metrics
│   ├── calibration.py         # Calibration metrics
│   └── visualization.py       # Visualization utils
│
├── utils/                      # Utilities
│   ├── logging.py             # Logging and experiment tracking
│   └── reproducibility.py     # Seed control
│
├── experiments/                # Experiment scripts
│   ├── configs/               # YAML configs
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
│
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── test_system.py             # Validation tests
```

## Hyperparameter Tuning

### Key Hyperparameters

1. **Thresholds** (tune on validation set):
   - `no_change`: Probability threshold for no-change (default: 0.5)
   - `change_confident`: Probability threshold for confident change (default: 0.7)
   - `log_likelihood`: Log-likelihood threshold for in-distribution (default: -10.0)
   - `uncertainty`: Epistemic uncertainty threshold (default: 0.3)

2. **Model Architecture**:
   - `backbone`: resnet18 (faster) vs resnet34 (more capacity)
   - `embedding_dim`: 64 vs 128
   - `n_gmm_components`: 2-5 for GMM
   - `n_flows`: 4-8 for normalizing flow

3. **Training**:
   - `learning_rate`: 1e-4 to 1e-3
   - `batch_size`: 8-32 (depends on GPU memory)
   - `loss.pos_weight`: 1.5-3.0 (handle class imbalance)

### Grid Search Example

```python
# Create multiple configs with different hyperparameters
for lr in [1e-4, 5e-4, 1e-3]:
    for embedding_dim in [64, 128]:
        # Create config file
        # Run training
        # Evaluate on validation set
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size`
   - Use `mixed_precision: true`
   - Reduce `patch_size`

2. **Density Model Not Fitting**:
   - Ensure enough change samples (`min_samples` in config)
   - Check dataset labels are correct
   - Try simpler model (GMM instead of Flow)

3. **Poor Performance**:
   - Check data normalization
   - Verify dataset splits (no leakage)
   - Tune class imbalance weight (`pos_weight`)
   - Increase training epochs

## Citation

If you use this code, please cite:

```bibtex
@misc{change_detection_ood_2026,
  title={Remote Sensing Change Detection with Out-of-Distribution Awareness},
  author={},
  year={2026},
  url={https://github.com/...}
}
```

## References

1. Daudt, R. C., Le Saux, B., & Boulch, A. (2018). "Fully Convolutional Siamese Networks for Change Detection." IGARSS 2018.

2. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017.

3. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). "Density Estimation using Real NVP." ICLR 2017.

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

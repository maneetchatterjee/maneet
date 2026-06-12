# Remote Sensing Change Detection with OOD Awareness

A research-grade change detection system for remote sensing that incorporates out-of-distribution detection and epistemic uncertainty estimation.

## Overview

This implementation follows the architectural lineage of:
- Daudt et al., "Urban Change Detection for Multispectral EO using CNNs" (IGARSS 2018)
- UnCRtainTS approach for uncertainty estimation (CVPR EarthVision 2023)

### Key Features

- **Siamese CNN Architecture**: Shared-weight ResNet-18/34 encoders for bi-temporal change detection
- **Change Embedding**: Latent representation of how changes occur
- **Density Modeling**: GMM and Normalizing Flow for in-distribution modeling
- **Deep Ensembles**: Epistemic uncertainty via model disagreement
- **Three-Way Decision**: no-change / change (confident) / change (OOD, abstain)

## Architecture

```
Input: (I_t, I_t') → Siamese ResNet Encoder
                    ↓
            Feature Difference: |f_t - f_t'|
                    ↓
            Global Average Pooling + MLP
                    ↓
            Change Embedding (z_Δ)
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
  Binary Classifier      Density Model
  (change/no-change)     (log-likelihood)
                    ↓
            Ensemble Uncertainty
                    ↓
            Three-Way Decision
```

## Installation

```bash
pip install -r requirements.txt
```

## Directory Structure

```
/models
    siamese_resnet.py          # Siamese ResNet backbone
    change_embedding.py        # Change embedding module
    density_models/
        gmm.py                 # Gaussian Mixture Model
        normalizing_flow.py    # RealNVP/MAF implementation
    ensemble.py                # Deep ensemble wrapper

/datasets
    oscd.py                    # OSCD dataset loader
    levir_cd.py                # LEVIR-CD dataset loader

/training
    trainer.py                 # Training loop with mixed precision
    losses.py                  # Loss functions

/evaluation
    metrics.py                 # Change detection metrics
    ood_metrics.py             # OOD detection metrics
    calibration.py             # Reliability metrics

/experiments
    configs/                   # YAML configuration files
    train.py                   # Training script
    evaluate.py                # Evaluation script

/utils
    logging.py                 # Logging utilities
    reproducibility.py         # Seed control and determinism
```

## Usage

### Training

```bash
python experiments/train.py --config experiments/configs/oscd_baseline.yaml
```

### Evaluation

```bash
python experiments/evaluate.py --config experiments/configs/oscd_baseline.yaml \
                               --checkpoint path/to/checkpoint.pth
```

### Configuration

Example configuration (`experiments/configs/oscd_baseline.yaml`):

```yaml
model:
  backbone: resnet18
  embedding_dim: 128
  density_model: gmm  # or normalizing_flow
  
ensemble:
  n_models: 3
  
training:
  batch_size: 16
  epochs: 100
  optimizer: adamw
  lr: 1e-4
  
thresholds:
  no_change: 0.5
  change_confident: 0.7
  log_likelihood: -10.0
  uncertainty: 0.3
```

## Datasets

### OSCD (Onera Satellite Change Detection)
- Sentinel-2 imagery
- Bi-temporal pairs
- Binary change labels
- Urban areas

### LEVIR-CD
- High-resolution imagery
- Building change detection

**Note**: Datasets must be downloaded separately and placed in the `data/` directory.

## Evaluation Metrics

### Change Detection
- Precision, Recall, F1
- Confusion matrices
- Per-class metrics

### OOD Detection
- AUROC (in-distribution vs OOD)
- Likelihood histograms
- Confidence calibration

### Reliability
- Coverage vs accuracy curves
- Risk-coverage plots
- Abstention rate vs error

## Ablation Studies

The codebase supports the following ablations:
1. No density model (baseline)
2. GMM vs Normalizing Flow
3. Single model vs ensemble
4. With vs without abstention

## Citation

If you use this code, please cite:

```bibtex
@misc{change_detection_ood_2026,
  title={Remote Sensing Change Detection with Out-of-Distribution Awareness},
  author={},
  year={2026}
}
```

## License

MIT License

## References

- Daudt, R. C., Le Saux, B., & Boulch, A. (2018). "Fully Convolutional Siamese Networks for Change Detection." IGARSS 2018.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017.

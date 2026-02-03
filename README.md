# E3GNN-XANES

An E(3)-Equivariant Graph Neural Network (GNN) implementation for predicting X-ray Absorption Near Edge Structure (XANES) spectra.

## Key Features

- **Physics-Informed Architecture**: Strictly enforces the physics of core-hole interactions.
- **E(3) Equivariance**: Built with `e3nn` to handle geometric tensors and maintain rotational invariance.
- **Multi-Scale Gaussian Basis**: Uses multiple widths of Radial Basis Functions (RBFs) to capture both sharp absorption edges and broad scattering oscillations.
- **Absorber-Query Attention**: Concentrates environmental context using the absorbing atom as the attention query.
- **Anisotropy-Aware Readout**: Preserves the magnitude of $l>0$ features (dipole/quadrupoles) to capture local crystal field splitting effects.

## Project Structure

```text
E3GNN_xanes/
├── src/
│   ├── model/
│   │   ├── basis.py      # Multi-scale Gaussian basis implementation
│   │   ├── layers.py     # Atomic embeddings and custom interaction blocks
│   │   ├── pooling.py    # Absorber-query attention mechanism
│   │   └── gnn.py        # Main XANES_E3GNN architecture
│   ├── loss.py           # Combined MSE + Gradient shape loss
│   └── train.py          # Training loop and utility functions
├── verify_model.py       # Smoke test using synthetic data
├── check_equivariance.py # Script to verify rotation invariance
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Installation

1. Create a conda environment:
   ```bash
   conda create -n gnn_xanes python=3.10
   conda activate gnn_xanes
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Verification
Run the verification script to ensure the model architecture and training loop are functioning correctly:
```bash
python verify_model.py
```
This will create a `best_model.pt` file upon completion.

### Symmetry Check
Verify that the model satisfies physical rotation invariance:
```bash
python check_equivariance.py
```

### Training
To train on custom data:
```python
import torch
from src import XANES_E3GNN, SpectrumLoss, run_training

# 1. Setup Model
model = XANES_E3GNN(
    num_layers=4,
    lmax=2,
    num_basis=128
)

# 2. Setup Loss
criterion = SpectrumLoss(lambda_grad=0.5)

# 3. Train
config = {
    'lr': 1e-3,
    'batch_size': 16,
    'epochs': 100,
    'criterion': criterion,
    'energy_grid': energy_grid,
    'save_path': 'xanes_model.pt'
}
run_training(model, train_dataset, val_dataset, config)
```

## Mathematical Framework

- **Scattering Propagator**:
  $m_{ij} = TP(h_j, Y(\hat{r}_{ij})) \cdot MLP(d_{ij})$
- **Reconstruction**:
  $\hat{\mu}(E) = B \cdot c_{basis}$
  Where $B$ is the multi-scale basis matrix and $c_{basis}$ are the predicted coefficients.
- **Loss**:
  $L = MSE(\mu, \hat{\mu}) + \lambda MSE(\nabla_E \mu, \nabla_E \hat{\mu})$

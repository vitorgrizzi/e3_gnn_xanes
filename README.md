# E3GNN-XANES

An E(3)-Equivariant Graph Neural Network (GNN) implementation for predicting X-ray Absorption Near Edge Structure (XANES) spectra.

## Key Features

- **Physics-Informed Architecture**: Strictly enforces the physics of core-hole interactions.
- **E(3) Equivariance**: Built with `e3nn` to handle geometric tensors and maintain rotational invariance.
- **PBC-Aware Periodicity**: Handles Periodic Boundary Conditions correctly via displacement vectors, allowing seamless training on bulk solids.
- **Multi-Site Handling**: Correctly processes structures with multiple absorber atoms, generating per-site spectral predictions.
- **Multi-Scale Gaussian Basis**: Uses multiple widths of Radial Basis Functions (RBFs) to capture both sharp absorption edges and broad scattering oscillations.
- **Absorber-Query Attention**: Concentrates environmental context using the absorbing atom as the attention query.

## Project Structure

```text
E3GNN_xanes/
├── src/
│   ├── data/
│   │   ├── assemble_dataset.py  # Parses FDMNES outputs into ASE SQLite DB
│   │   ├── dataset.py           # PyG dataset implementation (ASE DB backed)
│   │   ├── transforms.py        # Graph connectivity transforms
│   │   └── utils.py             # Spectral normalization & parsing utilities
│   ├── model/
│   │   ├── basis.py             # Multi-scale Gaussian basis
│   │   ├── layers.py            # Equivariant interaction blocks
│   │   ├── pooling.py           # Absorber-query attention
│   │   └── gnn.py               # Main PBC-aware architecture
│   ├── loss.py                  # Combined MSE + Gradient shape loss
│   └── train.py                 # Training entry point (Hydra/WandB)
├── configs/
│   └── config.yaml              # Hydra configuration for model & training
├── tests/
│   ├── test_data.py             # Verification for PBC & multi-site data
│   └── test_model.py            # Equivariance & forward pass tests
├── requirements.txt
└── README.md
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

## Workflow

### 1. Data Preparation
First, assemble your FDMNES calculation outputs into an ASE SQLite database:
```bash
python src/data/assemble_dataset.py
```
This utility recursively scans directories, extracts structural data (with PBC), and stores normalized spectra in `xanes_dataset.db`.

### 2. Verification
Ensure the data pipeline and model geometry are correct:
```bash
# Test the PBC dataset pipeline
python tests/test_data.py

# Test model forward pass & equivariance
python -m pytest tests/test_model.py
```

### 3. Training
Train the model using the Hydra-based entry point:
```bash
python src/train.py data.db_path=/path/to/xanes_dataset.db
```
Configuration settings (layers, hidden dims, learning rate, etc.) can be modified in `configs/config.yaml` or overridden via CLI.

## Mathematical Framework

**Periodic Interaction**:
The model computes true displacement vectors $\mathbf{r}_{ij}$ that wrap across cell boundaries:
```math
\mathbf{r}_{ij} = \mathbf{pos}_j - \mathbf{pos}_i + \mathbf{S}_{ij} \cdot \mathbf{h}
```
Where $\mathbf{S}_{ij}$ is the integer periodic shift and $\mathbf{h}$ is the lattice matrix.

**Spectral Reconstruction**:
```math
\hat{\mu}(E) = \sum_{k} c_k \cdot G_k(E)
```
Where $G_k$ are the multi-scale Gaussian basis functions and $c_k$ are the scalars predicted by the equivariant backbone.

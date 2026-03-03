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
│   │   ├── preprocess.py        # Data augmentation and energy grid processing
│   │   └── utils.py             # Spectral normalization & parsing utilities
│   ├── model/
│   │   ├── basis.py             # Multi-scale Gaussian basis + Sigmoid background
│   │   ├── layers.py            # Equivariant interaction blocks (Bessel/Gaussian)
│   │   ├── pooling.py           # Absorber-query attention
│   │   └── gnn.py               # Main PBC-aware architecture
│   ├── loss.py                  # Combined MSE + Gradient loss + Laplacian loss
│   ├── train.py                 # Training entry point (Hydra/WandB)
│   ├── inference.py             # Model loading and spectral prediction API
│   ├── visualization.py         # Plotting tools for model evaluation
│   ├── eval_validation.py       # Detailed post-training performance metrics
│   └── config.py                # Configuration schema and defaults
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

**Equivariant Message Passing**:
The interaction between atoms is modeled using an E(3)-equivariant Tensor Product (TP). The message $m_{ij}$ from neighbor $j$ to site $i$ is computed as:
```math
m_{ij} = \bigoplus_{l_{out}} \sum_{l_1, l_2} w_{l_1, l_2, l_{out}}(|\mathbf{r}_{ij}|) \left[ h_j^{(l_1)} \otimes Y^{(l_2)}(\hat{\mathbf{r}}_{ij}) \right]^{(l_{out})}
```
Where:
- $\otimes$ denotes the tensor product decomposed via Clebsch-Gordan coefficients.
- $h_j^{(l_1)}$ are the irrep-based node features of center $j$.
- $Y^{(l_2)}(\hat{\mathbf{r}}_{ij})$ are the spherical harmonics of the unit edge vector.
- $w(|\mathbf{r}_{ij}|)$ are weights predicted by an MLP from the radial basis.

The update rule for node $i$ includes gated activations and a residual connection:
```math
h_i^{(l+1)} = \text{Gate}\left( \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} m_{ij} \right) + \mathbf{W}_{res} h_i^{(l)}
```

**Spectral Reconstruction**:
```math
\hat{\mu}(E) = \sum_{k} c_k \cdot B_k(E)
```
Where $B_k$ are the basis functions (Multi-scale Gaussians + Global Sigmoid Background) and $c_k$ are the coefficients predicted by the graph readout.

**Loss Function**:
To ensure high-fidelity spectral shapes, we minimize a composite loss:
```math
\mathcal{L} = \mathcal{L}_{MSE} + \lambda_{grad} \mathcal{L}_{grad} + \lambda_{lap} \mathcal{L}_{lap}
```
Where:
- $\mathcal{L}_{MSE}$: Standard Mean Squared Error on intensity.
- $\mathcal{L}_{grad}$: MSE between the first derivatives (Spectral Gradient) of predicted and true spectra.
- $\mathcal{L}_{lap}$: MSE between the second derivatives (Spectral Laplacian) to ensure peak sharpness.

**Periodic Interaction**:
The model computes true displacement vectors $\mathbf{r}_{ij}$ that wrap across cell boundaries:
```math
\mathbf{r}_{ij} = \mathbf{pos}_j - \mathbf{pos}_i + \mathbf{S}_{ij} \cdot \mathbf{h}
```
Where $\mathbf{S}_{ij}$ is the integer periodic shift and $\mathbf{h}$ is the lattice matrix.

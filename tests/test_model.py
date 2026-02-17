import torch
from e3nn import o3


def test_model_forward(model, dataset):
    """Test forward pass of the model with PBC-aware data."""
    data = dataset[0]
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.batch = batch

    # Ensure edge_shift exists (create_dummy_data already provides it)
    assert hasattr(data, "edge_shift"), "Data must have edge_shift for PBC"

    # Forward pass
    coeffs = model(data)
    
    # Check output shape [Batch, NumBasis]
    assert coeffs.shape == (1, 16)  # num_basis=16
    
    # Check spectrum prediction
    energy_grid = torch.linspace(-10, 50, 50)
    spectra = model.predict_spectra(data, energy_grid)
    assert spectra.shape == (1, 50)


def test_equivariance(model, dataset):
    """Test E(3) equivariance (rotational invariance of the scalar output).
    
    When rotating positions, we must also rotate edge_shift to keep the
    geometry consistent.
    """
    data = dataset[0]
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.batch = batch
    
    # 1. Original prediction
    with torch.no_grad():
        coeffs_orig = model(data)
        
    # 2. Rotate positions AND edge_shift
    rot = o3.matrix_x(torch.pi / 2.0)
    data_rot = data.clone()
    data_rot.pos = torch.matmul(data.pos, rot.T)
    if hasattr(data_rot, "edge_shift") and data_rot.edge_shift is not None:
        data_rot.edge_shift = torch.matmul(data_rot.edge_shift, rot.T)
    
    # 3. Rotated prediction
    with torch.no_grad():
        coeffs_rot = model(data_rot)
        
    # Check that coefficients (scalars) are invariant
    assert torch.allclose(coeffs_orig, coeffs_rot, atol=1e-5), \
        f"Max diff: {(coeffs_orig - coeffs_rot).abs().max()}"

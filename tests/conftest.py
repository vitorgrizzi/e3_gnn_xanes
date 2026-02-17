import pytest
import torch
from src.data.dataset import create_dummy_data
from src.model import XANES_E3GNN

@pytest.fixture
def dataset():
    """Provides a synthetic dataset for model tests."""
    num_energy = 50
    # Use the new create_dummy_data which produces edge_shift and cell
    data_list, _ = create_dummy_data(
        num_graphs=2,
        num_energy_points=num_energy,
        emin=-10.0,
        emax=50.0
    )
    return data_list

@pytest.fixture
def model():
    """Creates a small XANES_E3GNN model for testing."""
    return XANES_E3GNN(
        max_z=100,
        num_layers=2,
        lmax=1,
        mul_0=8,
        mul_1=4,
        mul_2=2,
        r_max=5.0,
        num_basis=16,
        num_radial=5,
        basis_scales=[0.5, 1.0],
        emin=-10.0,
        emax=50.0
    )

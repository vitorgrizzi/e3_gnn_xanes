"""
Verify the model pipeline by running a short training loop on synthetic data.

Usage:
    python -m scripts.verify_model
    # or from project root:
    python scripts/verify_model.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import XANES_E3GNN, SpectrumLoss, run_training
from src.data.dataset import create_dummy_data


def verify():
    print("Creating dummy dataset...")
    dataset, energy_grid = create_dummy_data(20)
    
    print("Initializing Model...")
    model = XANES_E3GNN(
        max_z=100,
        num_layers=2,
        lmax=2,
        num_basis=32,
    )
    
    criterion = SpectrumLoss(lambda_grad=0.5)
    
    config = {
        'lr': 1e-3,
        'batch_size': 4,
        'epochs': 2,
        'criterion': criterion,
        'energy_grid': energy_grid,
        'save_path': 'best_model.pt',
    }
    
    print("Starting Dummy Training Loop...")
    try:
        run_training(model, dataset, dataset[:4], config)
        print("SUCCESS: Training loop ran without crashing.")
    except Exception as e:
        print(f"FAILED: Training loop crashed.")
        print(e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify()

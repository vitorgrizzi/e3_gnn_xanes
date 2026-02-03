import torch
import torch.nn as nn
from torch_geometric.data import Data
from src.model.gnn import XANES_E3GNN
from src.loss import SpectrumLoss
from src.train import run_training

def create_dummy_data(num_graphs=10):
    dataset = []
    energy_grid = torch.linspace(-10, 50, 100) # 100 points
    
    for _ in range(num_graphs):
        # 5 atoms
        num_nodes = 5
        pos = torch.randn(num_nodes, 3)
        z = torch.randint(1, 80, (num_nodes,))
        
        # Edges (fully connected for simplicity)
        # Create full adjacency
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        
        # Absorber mask: just pick the first atom
        absorber_mask = torch.zeros(num_nodes, dtype=torch.bool)
        absorber_mask[0] = True
        
        # Fake target spectrum: just a gaussian
        y_spectrum = torch.exp(-(energy_grid - 10)**2 / 20)
        
        data = Data(
            x=z.unsqueeze(1), # [N, 1] usually expected or z
            z=z,              # providing both for fallback compatibility
            pos=pos,
            edge_index=edge_index,
            absorber_mask=absorber_mask,
            y=y_spectrum.unsqueeze(0) # [1, 100]
        )
        dataset.append(data)
    
    return dataset, energy_grid

def verify():
    print("Creating dummy dataset...")
    dataset, energy_grid = create_dummy_data(20)
    
    print("Initializing Model...")
    model = XANES_E3GNN(
        max_z=100,
        num_layers=2, # Keep it small for fast checks
        lmax=2,
        num_basis=32 # Small basis
    )
    
    criterion = SpectrumLoss(lambda_grad=0.5)
    
    config = {
        'lr': 1e-3,
        'batch_size': 4,
        'epochs': 2,
        'criterion': criterion,
        'energy_grid': energy_grid,
        'save_path': 'best_model.pt'
    }
    
    print("Starting Dummy Training Loop...")
    try:
        run_training(model, dataset, dataset[:4], config)
        print("SUCCESS: Training loop ran without crashing.")
    except Exception as e:
        print(f"FAILED: Training loop match crashed.")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()

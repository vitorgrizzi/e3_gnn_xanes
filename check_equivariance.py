import torch
from torch_geometric.data import Data
from e3nn import o3
from src.model.gnn import XANES_E3GNN

def check_invariance():
    print("Checking Rotation Invariance...")
    
    # 1. Setup Data
    # 5 atoms, random positions
    pos = torch.randn(5, 3)
    z = torch.randint(1, 80, (5,))
    batch = torch.zeros(5, dtype=torch.long)
    
    # Fully connected edges
    row = torch.arange(5).repeat_interleave(5)
    col = torch.arange(5).repeat(5)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    absorber_mask = torch.zeros(5, dtype=torch.bool)
    absorber_mask[0] = True
    
    data = Data(x=z.unsqueeze(1), z=z, pos=pos, edge_index=edge_index, batch=batch, absorber_mask=absorber_mask)
    
    # 2. Setup Model
    model = XANES_E3GNN(max_z=100, num_layers=2, num_basis=32)
    model.eval()
    
    # 3. Predict Original
    energy_grid = torch.linspace(-10, 50, 50)
    with torch.no_grad():
        out_orig = model.predict_spectra(data, energy_grid)
        
    # 4. Rotate Data
    # Random rotation matrix
    R = o3.rand_matrix()
    pos_rot = torch.matmul(pos, R.T) # Rotate positions
    
    data_rot = Data(x=z.unsqueeze(1), z=z, pos=pos_rot, edge_index=edge_index, batch=batch, absorber_mask=absorber_mask)
    
    # 5. Predict Rotated
    with torch.no_grad():
        out_rot = model.predict_spectra(data_rot, energy_grid)
        
    # 6. Compare
    diff = (out_orig - out_rot).abs().max()
    print(f"Max difference after rotation: {diff.item():.6e}")
    
    if diff < 1e-4:
        print("PASS: Model is invariant to rotation.")
    else:
        print("FAIL: Model output changed significantly after rotation.")

if __name__ == "__main__":
    check_invariance()

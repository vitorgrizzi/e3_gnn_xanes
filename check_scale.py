import torch
from src.data.dataset import XANESDataset
from src.model import XANES_E3GNN

print("Loading Dataset...")
db = XANESDataset('data/processed_test', 'xanes_data.db', preprocess=False)

if len(db) > 0:
    y = db[0].y
    print(f"Data y shape: {y.shape}")
    print(f"Data y min / max / mean / std: {y.min().item():.2f} / {y.max().item():.2f} / {y.mean().item():.2f} / {y.std().item():.2f}")
    
#     print("\nChecking Model Forward Pass...")
#     model = XANES_E3GNN(
#         max_z=100, 
#         num_layers=4, 
#         lmax=2, 
#         mul_0=64, 
#         mul_1=32, 
#         mul_2=16,
#         r_max=5.0,
#         num_basis=128,
#         num_radial=10,
#         basis_scales=[0.1, 0.5, 1.0],
#         emin=-30, emax=100
#     )
    
#     # Needs batch handling
#     data = db[0]
#     batch = torch.zeros(data.num_nodes, dtype=torch.long)
#     data.batch = batch
    
#     energy_grid = torch.linspace(-30, 100, 150)
    
#     spectra_pred = model.predict_spectra(data, energy_grid)
    
#     print(f"Pred spectra shape: {spectra_pred.shape}")
#     print(f"Pred spectra min / max / mean / std: {spectra_pred.min().item():.2f} / {spectra_pred.max().item():.2f} / {spectra_pred.mean().item():.2f} / {spectra_pred.std().item():.2f}")
    
#     from src.loss import SpectrumLoss
#     criterion = SpectrumLoss()
#     loss, loss0, loss1 = criterion(spectra_pred, data.y.unsqueeze(0) if data.y.ndim == 1 else data.y, energy_grid)
#     print(f"Initial untrained loss: {loss.item():.2f} (Intensity={loss0.item():.2f}, Grad={loss1.item():.2f})")
# else:
#     print("Dataset empty.")

import torch
from tqdm import tqdm
from src.data.dataset import XANESDataset
from src.model import XANES_E3GNN

print("Loading Dataset...")
db = XANESDataset('.', processed_path='C:/Users/Vitor/Downloads/data_rmax5.0_e150.pt')

maxes = []
mins = []
means = []
for i in tqdm(range(len(db))):
    y = db[i].y
    maxes.append(y.max().item())
    mins.append(y.min().item())


print(f"Max y: {max(maxes):.2f}")
print(f"Min y: {min(mins):.2f}")
print(f'maxes mean {sum(maxes)/len(maxes):.2f}')
    
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

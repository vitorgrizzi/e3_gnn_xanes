import torch
from tqdm import tqdm
from src.data.dataset import XANESDataset
from src.model import XANES_E3GNN

print("Loading Dataset...")
db = XANESDataset('.', processed_path='C:/Users/Vitor/Downloads/data_rmax5.0_e150.pt')

maxes = []
mins = []
means = []
z_abs = []
for i in tqdm(range(len(db))):
    y = db[i].y
    maxes.append(y.max().item())
    mins.append(y.min().item())
    z_abs.append(torch.sum(db[i].absorber_mask).item())

print(z_abs)
print(f"Max y: {max(maxes):.2f}")
print(f"Min y: {min(mins):.2f}")
print(f'maxes mean {sum(maxes)/len(maxes):.2f}')
print(f'z_abs mean {sum(z_abs)/len(z_abs):.2f}')


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
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.optim as optim

def train_epoch(model, loader, optimizer, criterion, device, energy_grid):
    model.train()
    total_loss = 0
    total_mse = 0
    total_grad = 0
    
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Helper logic since our 'data' might be synthetic or loaded differently
        # Expectation: data.y is [Batch, N_E] true spectra
        
        spectra_pred = model.predict_spectra(data, energy_grid)
        
        # Compute loss
        # Check if we need to unsqueeze or match dims
        # spectra_pred: [Batch, N_E]
        # data.y: [Batch, N_E]
        
        loss, mse, grad_loss = criterion(spectra_pred, data.y, energy_grid)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_mse += mse.item() * data.num_graphs
        total_grad += grad_loss.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss / n, total_mse / n, total_grad / n

def validate(model, loader, criterion, device, energy_grid):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            spectra_pred = model.predict_spectra(data, energy_grid)
            loss, _, _ = criterion(spectra_pred, data.y, energy_grid)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def run_training(model, train_dataset, val_dataset, config):
    device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 1e-3))
    criterion = config['criterion']
    energy_grid = config['energy_grid'].to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 4), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 4))
    
    epochs = config.get('epochs', 10)
    for epoch in range(epochs):
        train_loss, mse, grad = train_epoch(model, train_loader, optimizer, criterion, device, energy_grid)
        val_loss = validate(model, val_loader, criterion, device, energy_grid)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} (MSE: {mse:.4f}, Grad: {grad:.4f}) | Val Loss: {val_loss:.4f}")

    # Save model if save_path is provided
    save_path = config.get('save_path')
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

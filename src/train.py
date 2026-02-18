import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
from src.model import XANES_E3GNN
from src.loss import SpectrumLoss
from src.data.dataset import XANESDataset


def train_epoch(model, loader, optimizer, criterion, device, energy_grid, grad_clip=None):
    model.train()
    total_loss = 0
    total_mse = 0
    total_grad = 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        
        spectra_pred = model.predict_spectra(data, energy_grid)
        loss, mse, grad_loss = criterion(spectra_pred, data.y, energy_grid)
        
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_mse += mse.item() * data.num_graphs
        total_grad += grad_loss.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss / n, total_mse / n, total_grad / n


def validate(model, loader, criterion, device, energy_grid):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_grad = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            spectra_pred = model.predict_spectra(data, energy_grid)
            loss, mse, grad_loss = criterion(spectra_pred, data.y, energy_grid)
            total_loss += loss.item() * data.num_graphs
            total_mse += mse.item() * data.num_graphs
            total_grad += grad_loss.item() * data.num_graphs
    n = len(loader.dataset)
    return total_loss / n, total_mse / n, total_grad / n


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Setup WandB
    if cfg.wandb.mode != 'disabled':
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode
        )

    # 2. Device
    if cfg.training.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.training.device)
    print(f"Using device: {device}")

    # 3. Data Loading
    root_path = hydra.utils.to_absolute_path(cfg.data.root)
    db_path = hydra.utils.to_absolute_path(cfg.data.db_path) if cfg.data.db_path else None

    dataset = XANESDataset(
        root=root_path,
        db_path=db_path,
        r_max=cfg.data.r_max,
        emin=cfg.model.emin,
        emax=cfg.model.emax,
        num_energy_points=cfg.model.num_energy_points,
        preprocess=cfg.data.get('preprocess', False),
    )
    
    # Split (random split)
    g = torch.Generator().manual_seed(42) # Set a seed for reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=g)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
    )
    
    # 4. Model
    model = XANES_E3GNN(
        max_z=cfg.model.max_z,
        num_layers=cfg.model.num_layers,
        lmax=cfg.model.lmax,
        mul_0=cfg.model.mul_0,
        mul_1=cfg.model.mul_1,
        mul_2=cfg.model.mul_2,
        r_max=cfg.model.r_max,
        num_basis=cfg.model.num_basis,
        num_radial=cfg.model.num_radial,
        basis_scales=cfg.model.basis_scales,
        emin=cfg.model.emin,
        emax=cfg.model.emax
    ).to(device)
    
    # 5. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=cfg.training.patience // 2, verbose=True
    )
    criterion = SpectrumLoss(lambda_grad=cfg.training.lambda_grad)
    energy_grid = torch.linspace(cfg.model.emin, cfg.model.emax, cfg.model.num_energy_points).to(device)
    
    # 6. Training Loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    save_path = hydra.utils.to_absolute_path(cfg.training.save_path) if cfg.training.save_path else None
    
    for epoch in range(cfg.training.epochs):
        train_loss, train_mse, train_grad = train_epoch(
            model, train_loader, optimizer, criterion, device, energy_grid, cfg.training.grad_clip
        )
        val_loss, val_mse, val_grad = validate(model, val_loader, criterion, device, energy_grid)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}"
        )
        
        if cfg.wandb.mode != 'disabled':
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "lr": current_lr
            })
            
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

if __name__ == "__main__":
    main()

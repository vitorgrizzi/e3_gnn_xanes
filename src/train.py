import os
import time
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
from src.visualization import generate_validation_plots
from src.inference import load_model


def get_gpu_memory():
    if torch.cuda.is_available():
        # returns memory in GB
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0

def validate(model, loader, criterion, device, energy_grid):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_grad = 0
    total_lap = 0

    # No need to compute gradients for validation
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            spectra_pred = model.predict_spectra(data, energy_grid)
            loss, mse, grad_loss, lap_loss = criterion(spectra_pred, data.y, energy_grid)
            total_loss += loss.item() * data.num_graphs
            total_mse += mse.item() * data.num_graphs
            total_grad += grad_loss.item() * data.num_graphs
            total_lap += lap_loss.item() * data.num_graphs
    n = len(loader.dataset)
    return total_loss / n, total_mse / n, total_grad / n, total_lap / n


def train_epoch(model, loader, optimizer, criterion, device, energy_grid, grad_clip=None):
    model.train()
    total_loss = 0
    total_mse = 0
    total_grad = 0
    total_lap = 0
    
    # Reset GPU memory stats to measure peak memory usage during current epoch only
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    for batch_data in tqdm(loader, desc="Training", leave=False):
        batch_data = batch_data.to(device)
        optimizer.zero_grad() # erases gradients from previous step (pytorch accumulates gradients by default)
        
        spectra_pred = model.predict_spectra(batch_data, energy_grid)
        loss, mse, grad_loss, lap_loss = criterion(spectra_pred, batch_data.y, energy_grid)
        
        # Backpropagate loss through computational graph to fill .grad attribute of model parameter tensors
        loss.backward() 

        # Gradient clipping to prevent exploding gradients
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update model parameters with the gradients stored in .grad attribute computed by backpropagation
        optimizer.step() 
        
        # The loss returned is the average over the batch (MSE with reduction='mean'), 
        # so we multiply by the number of graphs in the batch to get the total batch loss
        total_loss += loss.item() * batch_data.num_graphs
        total_mse += mse.item() * batch_data.num_graphs
        total_grad += grad_loss.item() * batch_data.num_graphs
        total_lap += lap_loss.item() * batch_data.num_graphs
        
    n = len(loader.dataset)
    return total_loss / n, total_mse / n, total_grad / n, total_lap / n


def run_training(model, train_loader, val_loader, config, model_config=None):
    """
    Run the full training loop with given loaders and config.
    
    Args:
        model: XANES_E3GNN model.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Dictionary or OmegaConf containing lr, epochs, criterion, energy_grid, etc.
        model_config: Dictionary containing model hyperparameters for self-contained checkpoints.
    """
    device = config.get('device', next(model.parameters()).device)

    # AdamW optimizer with optional weight decay (equivalent to L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.01))

    # Reduce LR by half if validation loss doesn't decrease for patience//2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['patience'] // 2
    )

    criterion = config['criterion']
    energy_grid = config['energy_grid']
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    start_epoch = 0
    save_path = config.get('save_path')
    log_path = config.get('log_path')
    load_path = config.get('load_path')

    # Load checkpoint if provided
    if load_path and os.path.exists(load_path):
        print(f"Loading checkpoint from {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        
        # Checkpoint is a dict with 'model_state_dict'
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Initialize local text log file if starting fresh
    if log_path and start_epoch == 0:
        with open(log_path, 'w') as f:
            f.write("=== XANES E3GNN Training Log ===\n")
            f.write(f"Device: {device}\n")
            f.write(f"Learning Rate: {config['lr']}\n")
            f.write(f"Max Epochs: {config['epochs']}\n")
            f.write(f"Patience: {config['patience']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'T MSE':<12} {'V MSE':<12} {'T Grad MSE':<12} {'V Grad MSE':<12} {'T Lap MSE':<12} {'V Lap MSE':<12} {'LR':<10} {'GPU (GB)':<10} {'Time (m)':<10}\n")
            f.write("-" * 148 + "\n")
    elif log_path and start_epoch > 0:
        with open(log_path, 'a') as f:
            f.write(f"--- Resumed training from epoch {start_epoch + 1} ---\n")

    # Training loop, each iteration is one epoch
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        train_loss, train_mse, train_grad, train_lap = train_epoch(
            model, train_loader, optimizer, criterion, device, energy_grid, config.get('grad_clip')
        ) 
        val_loss, val_mse, val_grad, val_lap = validate(model, val_loader, criterion, device, energy_grid)
        epoch_time_min = (time.time() - epoch_start) / 60

        scheduler.step(val_loss) # Reduce LR if `val_loss` doesn't decrease
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Time: {epoch_time_min:.2f}m | LR: {current_lr:.2e}"
        )
        
        # Log to wandb if active
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "train_grad": train_grad,
                "val_grad": val_grad,
                "train_lap": train_lap,
                "val_lap": val_lap,
                "lr": current_lr,
                "gpu_mem_gb": get_gpu_memory(),
                "epoch_time_min": epoch_time_min
            })
            
        # Log to local text file
        if log_path:
            gpu_mem = get_gpu_memory()
            with open(log_path, 'a') as f:
                f.write(
                    f"{epoch+1:<8} {train_loss:<12.4f} {val_loss:<12.4f} "
                    f"{train_mse:<12.4f} {val_mse:<12.4f} {train_grad:<12.4f} {val_grad:<12.4f} "
                    f"{train_lap:<12.4f} {val_lap:<12.4f} {current_lr:<10.2e} {gpu_mem:<10.2f} {epoch_time_min:<10.2f}\n"
                )
            
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if save_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': model_config
                }, save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break


def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Setup WandB
    if cfg.wandb.mode != 'disabled':
        # Safely attempt to login. Will look for WANDB_API_KEY env var
        # or you can pass it here: wandb.login(key="YOUR_KEY")
        wandb.login()
        
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
    db_path = hydra.utils.to_absolute_path(cfg.data.get('db_path')) if cfg.data.get('db_path') else None
    proc_path = hydra.utils.to_absolute_path(cfg.data.get('processed_path')) if cfg.data.get('processed_path') else None

    dataset = XANESDataset(
        root=root_path,
        db_path=db_path,
        processed_path=proc_path,
        r_max=cfg.data.r_max,
        emin=cfg.model.emin,
        emax=cfg.model.emax,
        num_energy_points=cfg.model.num_energy_points,
        preprocess=cfg.data.get('preprocess', False),
    )
    
    # Split
    g = torch.Generator().manual_seed(42)
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
        emax=cfg.model.emax,
        dropout=cfg.model.get('dropout', 0.1)
    ).to(device)
    
    # 5. Optimization & Criterion
    # Setup Loss Function
    lambda_grad = cfg.training.get('lambda_grad', 0.5)
    lambda_lap = cfg.training.get('lambda_lap', 0.0)
    criterion = SpectrumLoss(lambda_grad=lambda_grad, lambda_lap=lambda_lap)
    energy_grid = torch.linspace(cfg.model.emin, cfg.model.emax, cfg.model.num_energy_points).to(device)
    
    config = {
        'lr': cfg.training.lr,
        'batch_size': cfg.training.batch_size,
        'epochs': cfg.training.epochs,
        'criterion': criterion,
        'energy_grid': energy_grid,
        'save_path': hydra.utils.to_absolute_path(cfg.training.save_path) if cfg.training.save_path else None,
        'load_path': hydra.utils.to_absolute_path(cfg.training.load_path) if cfg.training.get('load_path') else None,
        'log_path': hydra.utils.to_absolute_path(cfg.training.log_path) if cfg.training.get('log_path') else None,
        'patience': cfg.training.patience,
        'grad_clip': cfg.training.grad_clip,
        'weight_decay': cfg.training.get('weight_decay', 0.01),
        'device': device
    }
    
    # 6. Run Training
    run_training(model, train_loader, val_loader, config, model_config=OmegaConf.to_container(cfg.model, resolve=True))

    # 7. Post-training evaluation plots
    num_samples = cfg.training.get('num_val_samples', 0)
    if num_samples > 0 and config['save_path'] and os.path.exists(config['save_path']):
        print(f"\nTraining finished. Generating {num_samples} validation plots...")
        try:
            # Reload the best model to ensure we plot the best results
            best_model, _ = load_model(config['save_path'])
            best_model = best_model.to(device)
            
            output_dir = "validation_plots"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "best_model_val_samples.png")
            
            generate_validation_plots(
                model=best_model,
                val_dataset=val_dataset,
                energy_grid=energy_grid,
                num_samples=num_samples,
                output_path=output_path,
                device=device
            )
        except Exception as e:
            print(f"Warning: Could not generate validation plots: {e}")

if __name__ == "__main__":
    import argparse
    from hydra import initialize_config_dir, compose
    
    parser = argparse.ArgumentParser(description="Train XANES E3GNN model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml", 
        help="Path to the config.yaml file."
    )
    args = parser.parse_args()
    
    # 1. Resolve paths
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).replace(".yaml", "").replace(".yml", "")
    
    # 2. Initialize Hydra from the directory of the provided config file
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        main(cfg)

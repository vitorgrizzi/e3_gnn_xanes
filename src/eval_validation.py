import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from omegaconf import OmegaConf
from torch_geometric.data import DataLoader
from src.inference import load_model
from src.data.dataset import XANESDataset
from src.visualization import generate_validation_plots

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on random validation example(s).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file (if not in checkpoint)")
    parser.add_argument("--db", type=str, help="Override database path if different from config")
    parser.add_argument("--output", type=str, default="val_evaluation.png", help="Where to save the plot")
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of random samples to draw")
    
    args = parser.parse_args()
    
    # 1. Load Model and Config
    model, cfg = load_model(args.checkpoint, args.config)
    device = next(model.parameters()).device
    
    # 2. Setup Dataset
    root_path = cfg.data.root
    db_path = args.db if args.db else cfg.data.db_path
    
    if db_path is None:
         db_path = "xanes_dataset.db"
         print(f"Warning: db_path not found in config. Defaulting to {db_path}")

    print(f"Loading dataset from {db_path}...")
    dataset = XANESDataset(
        root=root_path,
        db_path=db_path,
        r_max=cfg.data.r_max,
        emin=cfg.model.emin,
        emax=cfg.model.emax,
        num_energy_points=cfg.model.num_energy_points,
        preprocess=False
    )
    
    g = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=g)
    
    print(f"Validation set size: {len(val_dataset)}")
    
    if len(val_dataset) == 0:
        print("Error: Validation set is empty.")
        return

    # 3, 4 & 5. Generate plots using shared utility
    energy_grid = torch.linspace(cfg.model.emin, cfg.model.emax, cfg.model.num_energy_points).to(device)
    
    generate_validation_plots(
        model=model,
        val_dataset=val_dataset,
        energy_grid=energy_grid,
        num_samples=args.num_samples,
        output_path=args.output,
        device=device
    )

if __name__ == "__main__":
    main()

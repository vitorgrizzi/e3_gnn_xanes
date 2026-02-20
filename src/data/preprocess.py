"""
Script to manually trigger dataset preprocessing.

Usage:
    python src/data/preprocess.py data.db_path=/path/to/db [other options]
"""
import argparse
import os
import torch
from omegaconf import OmegaConf
from src.data.dataset import XANESDataset

def main():
    # 1. Setup Parser
    parser = argparse.ArgumentParser(description="Preprocess XANES ASE database into PyG graphs.")
    parser.add_argument("db_path", type=str, help="Path to the raw ASE SQLite database (.db)")
    parser.add_argument("--root", type=str, default=None, help="Root directory for processed files (defaults to config.yaml value)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to project config file")
    parser.add_argument("--rmax", type=float, default=None, help="Cutoff radius (Å)")
    parser.add_argument("--preprocess", action="store_true", default=True, help="Force reprocessing.")

    args = parser.parse_args()

    # 2. Load Config for defaults
    if os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        print(f"Warning: Config file {args.config} not found. using library defaults.")
        cfg = OmegaConf.create({
            "data": {"root": "data/processed", "r_max": 5.0},
            "model": {"emin": -30.0, "emax": 100.0, "num_energy_points": 150}
        })

    # 3. Resolve Paths and Params
    db_path = os.path.abspath(args.db_path)
    root_path = os.path.abspath(args.root if args.root else cfg.data.root)
    r_max = args.rmax if args.rmax else cfg.data.r_max
    
    print(f"--- Preprocessing Dataset ---")
    print(f"DB Path:      {db_path}")
    print(f"Root Path:    {root_path}")
    print(f"R Max:        {r_max}")
    print(f"Energy Grid:  {cfg.model.emin} to {cfg.model.emax} ({cfg.model.num_energy_points} pts)")
    print("-" * 30)

    # 4. Trigger Dataset Creation
    dataset = XANESDataset(
        root=root_path,
        db_path=db_path,
        r_max=r_max,
        emin=cfg.model.emin,
        emax=cfg.model.emax,
        num_energy_points=cfg.model.num_energy_points,
        preprocess=args.preprocess
    )
    
    print(f"\nSuccessfully processed {len(dataset)} graphs.")
    print(f"Saved to: {dataset.processed_paths[0]}")

if __name__ == "__main__":
    main()

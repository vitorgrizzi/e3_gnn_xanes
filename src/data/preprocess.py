"""
Script to manually trigger dataset preprocessing.

Usage:
    python src/data/preprocess.py data.db_path=/path/to/db [other options]
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import XANESDataset
import os

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    root_path = hydra.utils.to_absolute_path(cfg.data.root)
    db_path = hydra.utils.to_absolute_path(cfg.data.db_path) if cfg.data.db_path else None
    
    if db_path is None:
        print("Error: data.db_path is not set.")
        return

    print(f"Preprocessing dataset from {db_path}...")
    print(f"Root directory: {root_path}")
    
    # Force preprocessing by setting preprocess=True
    dataset = XANESDataset(
        root=root_path,
        db_path=db_path,
        r_max=cfg.data.r_max,
        emin=cfg.model.emin,
        emax=cfg.model.emax,
        num_energy_points=cfg.model.num_energy_points,
        preprocess=True 
    )
    
    print(f"Successfully processed {len(dataset)} graphs.")
    print(f"Saved to {dataset.processed_paths[0]}")

if __name__ == "__main__":
    main()

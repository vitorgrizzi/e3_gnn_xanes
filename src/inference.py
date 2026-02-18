"""
Inference script for predicting XANES spectra from structure files.

Usage:
    python src/inference.py --checkpoint best_model.pt --structure structure.cif
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from src.model import XANES_E3GNN
from src.data import atoms_to_graph
from omegaconf import OmegaConf

def load_model(checkpoint_path, config_path=None):
    """
    Loads the trained model and its configuration.
    
    If config_path is not provided, it looks for it in the same directory as the checkpoint.
    """
    if config_path is None:
        # Default to standard config location or look for a .hydra folder if it exists
        config_path = "configs/config.yaml" 
    
    cfg = OmegaConf.load(config_path)
    
    # Initialize model with config parameters
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
    )
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, cfg

def predict(model, atoms, r_max):
    """Predicts the spectrum for an ASE Atoms object."""
    # Convert Atoms to PyG Data
    data = atoms_to_graph(atoms, r_max=r_max)
    
    # Add dummy batch attribute
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    with torch.no_grad():
        # The model uses predict_spectra method
        # We need the energy grid from the model's basis settings
        energy_grid = torch.linspace(model.basis.emin, model.basis.emax, 150) # Matching dataset default
        spectrum = model.predict_spectra(data, energy_grid)
    
    return energy_grid.numpy(), spectrum.squeeze().numpy()

def main():
    parser = argparse.ArgumentParser(description="Predict XANES spectrum for a structure.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--structure", type=str, required=True, help="Path to structure file (CIF, POSCAR, XYZ)")
    parser.add_argument("--output", type=str, default="prediction.png", help="Path to save the plot")
    parser.add_argument("--tag_absorbers", type=int, nargs='+', help="Indices of absorber atoms (set tag=1)")
    
    args = parser.parse_args()
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, args.config)
    
    # 2. Prepare Structure
    print(f"Reading structure from {args.structure}...")
    atoms = read(args.structure)
    
    # Update tags if provided manually, otherwise assume the file has them
    if args.tag_absorbers is not None:
        tags = np.zeros(len(atoms), dtype=int)
        tags[args.tag_absorbers] = 1
        atoms.set_tags(tags)
    
    # 3. Predict
    print("Running inference...")
    try:
        energies, intensities = predict(model, atoms, r_max=cfg.model.r_max)
        
        # 4. Save/Plot
        plt.figure(figsize=(8, 5))
        plt.plot(energies, intensities, lw=2, color='navy')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Predicted XANES Spectrum: {os.path.basename(args.structure)}")
        plt.grid(alpha=0.3)
        plt.savefig(args.output)
        print(f"Prediction saved to {args.output}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Tip: Make sure the structure has absorber atoms tagged (tag=1) or use --tag_absorbers.")

if __name__ == "__main__":
    main()

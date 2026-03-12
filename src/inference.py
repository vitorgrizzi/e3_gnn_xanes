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


def load_model(checkpoint_path):
    """
    Loads the trained model and its configuration directly from a self-contained checkpoint.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Check if this is a training checkpoint (dict) or a raw state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model_config = checkpoint.get('model_config')
        data_config = checkpoint.get('data_config')
    else:
        raise ValueError("Provided checkpoint is not a valid training dictionary or is missing configuration blocks.")

    print("Using model and data configuration from checkpoint.")
    cfg = {"model": model_config, "data": data_config}
    
    # Initialize model with config parameters
    model = XANES_E3GNN(
        max_z=model_config['max_z'],
        num_layers=model_config['num_layers'],
        lmax=model_config['lmax'],
        mul_0=model_config['mul_0'],
        mul_1=model_config['mul_1'],
        mul_2=model_config['mul_2'],
        r_max=data_config['r_max'],
        num_basis=model_config['num_basis'],
        num_radial=model_config['num_radial'],
        radial_basis_type=model_config['radial_basis_type'],
        basis_scales=model_config['basis_scales'],
        emin=model_config['emin'],
        emax=model_config['emax'],
        dropout=model_config['dropout'],
        global_bg=model_config['global_bg']
    )
    
    model.load_state_dict(state_dict)
    model.eval() # Set model to evaluation mode
    
    return model, model_config, data_config

def predict(model, atoms, r_max, num_energy_points=100):
    """Predicts the spectrum for an ASE Atoms object."""
    # Convert Atoms to PyG Data
    data = atoms_to_graph(atoms, r_max=r_max)
    
    # Add dummy batch attribute
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    with torch.no_grad():
        # We use the energy grid parameters from the model's basis settings
        energy_grid = torch.linspace(
            model.basis.emin, 
            model.basis.emax, 
            num_energy_points
        ) 
        spectrum = model.predict_spectra(data, energy_grid)
    
    return energy_grid.numpy(), spectrum.squeeze().numpy()

def main():
    parser = argparse.ArgumentParser(description="Predict XANES spectrum for a structure.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--structure", type=str, required=True, help="Path to structure file (CIF, POSCAR, XYZ)")
    parser.add_argument("--output", type=str, default="prediction.png", help="Path to save the plot")
    parser.add_argument("--tag_absorbers", type=int, nargs='+', help="Specific indices of absorber atoms (set tag=1)")
    parser.add_argument("--absorber_z", type=int, help="Atomic number (Z) of absorbers to auto-tag")
    
    args = parser.parse_args()
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model, model_config, data_config = load_model(args.checkpoint)
    
    # Prepare Structure
    print(f"Reading structure from {args.structure}...")
    atoms = read(args.structure)
    
    tags = np.zeros(len(atoms), dtype=int)
    
    # Option A: Automatic tagging by atomic number (Z)
    if args.absorber_z is not None:
        z_array = atoms.get_atomic_numbers()
        indices = np.where(z_array == args.absorber_z)[0]
        if len(indices) == 0:
            print(f"Warning: No atoms with Z={args.absorber_z} found in structure.")
        else:
            print(f"Auto-tagged {len(indices)} atoms with Z={args.absorber_z} as absorbers.")
            tags[indices] = 1
            
    # Option B: Manual tagging by index (overwrites/adds to Z-tagging)
    if args.tag_absorbers is not None:
        tags[args.tag_absorbers] = 1

    # Apply tags if any were specified
    if args.absorber_z is not None or args.tag_absorbers is not None:
        atoms.set_tags(tags)
    
    # Predict
    print("Running inference...")
    try:
        energies, intensities = predict(
            model, 
            atoms, 
            r_max=data_config['r_max'], 
            num_energy_points=model_config['num_energy_points']
        )
        
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
        print("Tip: Make sure the structure has absorber atoms tagged (tag=1) or use --absorber_z or --tag_absorbers.")

if __name__ == "__main__":
    main()

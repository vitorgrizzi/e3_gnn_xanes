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
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        saved_config = checkpoint.get('model_config')
    else:
        state_dict = checkpoint
        saved_config = None

    if config_path is None and saved_config is None:
        config_path = 'configs/config.yaml'

    if saved_config is not None:
        print('Using model configuration from checkpoint.')
        cfg_model = OmegaConf.create(saved_config)
    else:
        print(f'Loading model configuration from {config_path}...')
        cfg = OmegaConf.load(config_path)
        cfg_model = cfg.model

    model = XANES_E3GNN(
        max_z=cfg_model.max_z,
        num_layers=cfg_model.num_layers,
        lmax=cfg_model.lmax,
        mul_0=cfg_model.mul_0,
        mul_1=cfg_model.mul_1,
        mul_2=cfg_model.mul_2,
        r_max=cfg_model.r_max,
        num_basis=cfg_model.num_basis,
        num_radial=cfg_model.num_radial,
        radial_basis_type=cfg_model.get('radial_basis_type', 'bessel'),
        basis_scales=cfg_model.basis_scales,
        emin=cfg_model.emin,
        emax=cfg_model.emax,
        dropout=cfg_model.get('dropout', 0.1),
        global_bg=cfg_model.get('global_bg', True),
        basis_focus_energy=cfg_model.get('basis_focus_energy', 15.0),
        basis_focus_left_width_ratio=cfg_model.get('basis_focus_left_width_ratio', 0.05),
        basis_focus_right_width_ratio=cfg_model.get('basis_focus_right_width_ratio', 0.25),
        basis_min_uniform_weight=cfg_model.get('basis_min_uniform_weight', 0.0),
        basis_max_uniform_weight=cfg_model.get('basis_max_uniform_weight', 0.9),
        basis_flatten_exponent=cfg_model.get('basis_flatten_exponent', 1.0),
        basis_cdf_resolution=cfg_model.get('basis_cdf_resolution', 4096),
    )

    model.load_state_dict(state_dict)
    model.eval()

    if saved_config is not None:
        full_cfg = OmegaConf.create({'model': saved_config})
    else:
        full_cfg = cfg

    return model, full_cfg


def predict(model, atoms, r_max, num_energy_points=100):
    """Predicts the spectrum for an ASE Atoms object."""
    data = atoms_to_graph(atoms, r_max=r_max)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    with torch.no_grad():
        energy_grid = torch.linspace(
            model.basis.emin,
            model.basis.emax,
            num_energy_points,
        )
        spectrum = model.predict_spectra(data, energy_grid)

    return energy_grid.numpy(), spectrum.squeeze().numpy()


def main():
    parser = argparse.ArgumentParser(description='Predict XANES spectrum for a structure.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument('--structure', type=str, required=True, help='Path to structure file (CIF, POSCAR, XYZ)')
    parser.add_argument('--output', type=str, default='prediction.png', help='Path to save the plot')
    parser.add_argument('--tag_absorbers', type=int, nargs='+', help='Specific indices of absorber atoms (set tag=1)')
    parser.add_argument('--absorber_z', type=int, help='Atomic number (Z) of absorbers to auto-tag')

    args = parser.parse_args()

    print(f'Loading model from {args.checkpoint}...')
    model, cfg = load_model(args.checkpoint, args.config)

    print(f'Reading structure from {args.structure}...')
    atoms = read(args.structure)

    tags = np.zeros(len(atoms), dtype=int)

    if args.absorber_z is not None:
        z_array = atoms.get_atomic_numbers()
        indices = np.where(z_array == args.absorber_z)[0]
        if len(indices) == 0:
            print(f'Warning: No atoms with Z={args.absorber_z} found in structure.')
        else:
            print(f'Auto-tagged {len(indices)} atoms with Z={args.absorber_z} as absorbers.')
            tags[indices] = 1

    if args.tag_absorbers is not None:
        tags[args.tag_absorbers] = 1

    if args.absorber_z is not None or args.tag_absorbers is not None:
        atoms.set_tags(tags)

    print('Running inference...')
    try:
        energies, intensities = predict(
            model,
            atoms,
            r_max=cfg.model.r_max,
            num_energy_points=cfg.model.num_energy_points,
        )

        plt.figure(figsize=(8, 5))
        plt.plot(energies, intensities, lw=2, color='navy')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(f'Predicted XANES Spectrum: {os.path.basename(args.structure)}')
        plt.grid(alpha=0.3)
        plt.savefig(args.output)
        print(f'Prediction saved to {args.output}')

    except ValueError as e:
        print(f'Error: {e}')
        print('Tip: Make sure the structure has absorber atoms tagged (tag=1) or use --absorber_z or --tag_absorbers.')


if __name__ == '__main__':
    main()
from ase import Atoms
from pathlib import Path
import numpy as np
from ase.io import read

def normalize_spectrum(conv_file: Path | str,
                       pre_edge_width: float=20.0,
                       post_edge_width: float=50.0,
                       ) -> tuple[np.ndarray, np.ndarray]:
    energy_xas = np.loadtxt(conv_file, skiprows=1) # (N,2) array

    E = energy_xas[:, 0].astype(float)
    mu = energy_xas[:, 1].astype(float)

    # Finding edge energy E0 (onset of absorption) if file doesn't set 0 as reference
    if np.min(np.abs(E)) > 5.0:
        dmu_dE = np.gradient(mu, E)
        E0 = E[np.argmax(dmu_dE)]
    else:
        E0 = 0

    # Finding pre- and post-edge masks
    pre_mask = E <= (E0 - pre_edge_width)
    post_mask = E >= (E0 + post_edge_width)

    # Doing linear fits μ ~ m*E + b
    m_pre, b_pre = np.polyfit(E[pre_mask], mu[pre_mask], 1)
    m_post, b_post = np.polyfit(E[post_mask], mu[post_mask], 1)

    # Subtract pre-edge to shift the pre_line to mu = 0
    pre_line = m_pre*E + b_pre
    mu_corr = mu - pre_line

    # Computing normalized mu
    step = (m_post*E0 + b_post) - (m_pre*E0 + b_pre)
    mu_norm = mu_corr / step

    return np.column_stack([E, mu_norm]), energy_xas


def extract_conv(fdmnes_output_dir: Path | str) -> np.ndarray:
    if not isinstance(fdmnes_output_dir, Path):
        fdmnes_output_dir = Path(fdmnes_output_dir)

    energy_xas = {}
    for i, conv_file in enumerate(fdmnes_output_dir.glob('*conv.txt')):
        energy_xas[i] = np.loadtxt(conv_file, skiprows=1) # (N,2) array

    return energy_xas
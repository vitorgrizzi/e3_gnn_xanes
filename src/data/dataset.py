"""
Dataset classes for XANES spectra prediction.

Reads structures from an ASE SQLite database (produced by assemble_dataset.py),
builds PBC-aware graphs via ASE neighbor lists, and produces one PyG Data
object per absorber site.
"""
import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from ase.db import connect
from ase.neighborlist import neighbor_list
from tqdm import tqdm


class XANESDataset(InMemoryDataset):
    """
    In-memory dataset for XANES spectra prediction.

    Reads from an ASE SQLite database where each row contains:
      - An ``Atoms`` object with PBC, cell, and fractional coordinates.
      - Absorber atoms tagged with ``tag == 1``.
      - XANES spectrum stored in ``row.data['xanes']`` as an (N, 2) array.

    For structures with *multiple* absorber atoms, this dataset creates
    **one graph per absorber site** (same structure, different absorber mask)
    so the model predicts a per-site spectrum.

    Each ``Data`` object carries:
        * ``z``             – (N,) atomic numbers
        * ``pos``           – (N, 3) Cartesian positions
        * ``cell``          – (3, 3) lattice matrix (row-vector convention)
        * ``edge_index``    – (2, E) directed edges
        * ``edge_shift``    – (E, 3) Cartesian PBC shift for each edge
        * ``absorber_mask`` – (N,) bool, True for the single absorber site
        * ``y``             – (1, N_E) interpolated XANES spectrum
    """

    def __init__(
        self,
        root: str,
        db_path: str | None = None,
        r_max: float = 5.0,
        emin: float = -30.0,
        emax: float = 100.0,
        num_energy_points: int = 150,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory where processed PyG files are cached.
            db_path: Path to the ASE SQLite database file.  If *None*, the
                     dataset assumes it has already been processed and loads
                     from ``root/processed/``.
            r_max: Cutoff radius (Å) for graph connectivity.
            emin / emax / num_energy_points: Target energy grid for spectrum
                interpolation.
        """
        self.db_path = db_path
        self.r_max = r_max
        self.emin = emin
        self.emax = emax
        self.num_energy_points = num_energy_points
        self.target_energy_grid = torch.linspace(emin, emax, num_energy_points)
        # super().__init__ calls process() if needed
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    # PyG boilerplate
    # ------------------------------------------------------------------
    @property
    def raw_file_names(self):
        if self.db_path is not None:
            return [os.path.basename(self.db_path)]
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    # ------------------------------------------------------------------
    # Heavy lifting
    # ------------------------------------------------------------------
    def process(self):
        if self.db_path is None:
            return

        data_list = []
        target_e = self.target_energy_grid.numpy()

        with connect(self.db_path) as db:
            n_rows = db.count()
            for row in tqdm(db.select(), total=n_rows, desc="Processing DB"):
                atoms = row.toatoms()

                # --- Structural tensors ---
                z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
                pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
                cell = torch.tensor(
                    np.array(atoms.get_cell()), dtype=torch.float
                )  # (3, 3)

                # --- PBC-aware neighbour list ---
                # 'i' source, 'j' destination, 'S' shift in fractional coords,
                # 'D' Cartesian distance vector  (D = pos[j]-pos[i]+S@cell)
                idx_i, idx_j, S, D = neighbor_list(
                    "ijSD", atoms, cutoff=self.r_max
                )

                edge_index = torch.tensor(
                    np.stack([idx_i, idx_j]), dtype=torch.long
                )  # (2, E)
                edge_shift = torch.tensor(
                    S @ np.array(atoms.get_cell()), dtype=torch.float
                )  # (E, 3)

                # --- Spectrum ---
                raw_spectrum = np.array(row.data.get("xanes"))
                if raw_spectrum is None or raw_spectrum.ndim != 2:
                    print(
                        f"  ⚠  Skipping row id={row.id}: missing / invalid "
                        f"spectrum (shape={getattr(raw_spectrum, 'shape', None)})"
                    )
                    continue

                sort_idx = np.argsort(raw_spectrum[:, 0])
                raw_e = raw_spectrum[sort_idx, 0]
                raw_y = raw_spectrum[sort_idx, 1]
                interp_y = np.interp(target_e, raw_e, raw_y)
                y = torch.tensor(interp_y, dtype=torch.float).unsqueeze(0)  # (1, N_E)

                # --- Absorber sites ---
                tags = atoms.get_tags()
                absorber_indices = np.where(tags == 1)[0]

                if len(absorber_indices) == 0:
                    print(
                        f"  ⚠  Skipping row id={row.id}: no absorber tag found"
                    )
                    continue

                # One Data object per absorber site
                for abs_idx in absorber_indices:
                    absorber_mask = torch.zeros(len(z), dtype=torch.bool)
                    absorber_mask[abs_idx] = True

                    data = Data(
                        z=z,
                        pos=pos,
                        cell=cell,
                        edge_index=edge_index,
                        edge_shift=edge_shift,
                        absorber_mask=absorber_mask,
                        y=y,
                    )

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

        print(f"Processed {len(data_list)} graphs from {self.db_path}")
        self.save(data_list, self.processed_paths[0])


# ----------------------------------------------------------------------
# Synthetic data for quick testing / debugging
# ----------------------------------------------------------------------
def create_dummy_data(
    num_graphs: int = 10,
    num_energy_points: int = 100,
    emin: float = -10.0,
    emax: float = 50.0,
):
    """
    Generate synthetic graph data that mirrors the schema produced by
    ``XANESDataset`` (including ``cell`` and ``edge_shift``).

    Returns:
        (list[Data], Tensor): List of Data objects and the energy grid.
    """
    dataset = []
    energy_grid = torch.linspace(emin, emax, num_energy_points)

    for _ in range(num_graphs):
        num_nodes = 5
        # Random cubic cell 5–10 Å
        a = 5.0 + 5.0 * torch.rand(1).item()
        cell = torch.eye(3) * a
        pos = torch.rand(num_nodes, 3) * a
        z = torch.randint(1, 80, (num_nodes,))

        # Fully connected edges (no self-loops) — simple stand-in
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_shift = torch.zeros(edge_index.size(1), 3)

        # Absorber mask: first atom
        absorber_mask = torch.zeros(num_nodes, dtype=torch.bool)
        absorber_mask[0] = True

        # Fake spectrum: Gaussian peak at 10 eV
        y_spectrum = torch.exp(-((energy_grid - 10) ** 2) / 20).unsqueeze(0)

        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            edge_shift=edge_shift,
            absorber_mask=absorber_mask,
            y=y_spectrum,
        )
        dataset.append(data)

    return dataset, energy_grid

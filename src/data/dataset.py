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

    For structures with multiple absorber atoms, this dataset creates
    one graph per absorber site (same structure, different absorber mask)
    so the model predicts a per-site spectrum.

    Each ``Data`` object carries:
        * ``z``               (N,) atomic numbers
        * ``pos``             (N, 3) Cartesian positions
        * ``cell``            (3, 3) lattice matrix (row-vector convention)
        * ``edge_index``      (2, E) directed edges
        * ``edge_shift``      (E, 3) Cartesian PBC shift for each edge
        * ``absorber_mask``   (N,) bool, True for the single absorber site
        * ``y``               (1, N_E) interpolated XANES spectrum
    """
    
    def __init__(
        self,
        root: str,
        db_path: str | None = None,
        r_max: float = 5.0,
        emin: float = -30.0,
        emax: float = 100.0,
        num_energy_points: int = 150,
        preprocess: bool = False,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory where processed PyG files are cached.
            db_path: Path to the ASE SQLite database file.
            r_max: Cutoff radius (Å) for graph connectivity.
            emin / emax / num_energy_points: Target energy grid for spectrum
                interpolation.
            preprocess: If True, forces reprocessing of the dataset even if
                        processed files exist.
        """
        self.db_path = db_path
        self.r_max = r_max
        self.emin = emin
        self.emax = emax
        self.num_energy_points = num_energy_points
        self.target_energy_grid = torch.linspace(emin, emax, num_energy_points)
        
        # Super init triggers processing if files are missing
        # We want to control this.
        super().__init__(root, transform, pre_transform)
        
        if preprocess and self.db_path is not None:
             # Force re-processing
             if os.path.exists(self.processed_paths[0]):
                 print(f"Preprocess=True: Deleting {self.processed_paths[0]} to force rebuild.")
                 os.remove(self.processed_paths[0])
             self.process()
        
        # Ensure we have data
        if not os.path.exists(self.processed_paths[0]) and self.db_path is not None:
             print("Processed files not found. Processing...")
             self.process()
             
        if os.path.exists(self.processed_paths[0]):
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
    # Processing the database and turning it into a graph
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
                
                # Extract target spectrum
                raw_spectrum = np.array(row.data.get("xanes"))
                if raw_spectrum is None or raw_spectrum.ndim != 2:
                    print(f"Skipping row id={row.id}: missing / invalid spectrum")
                    continue

                sort_idx = np.argsort(raw_spectrum[:, 0])
                raw_e = raw_spectrum[sort_idx, 0]
                raw_y = raw_spectrum[sort_idx, 1]
                interp_y = np.interp(target_e, raw_e, raw_y)
                y = torch.tensor(interp_y, dtype=torch.float).unsqueeze(0)  # (1, N_E)

                # Use helper to build graph
                try:
                    data = atoms_to_graph(atoms, r_max=self.r_max)
                    data.y = y # Add target
                    
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                        
                    data_list.append(data)
                except ValueError as e:
                    print(f"Skipping row id={row.id}: {e}")
                    continue

        print(f"Processed {len(data_list)} graphs from {self.db_path}")
        self.save(data_list, self.processed_paths[0])


def atoms_to_graph(atoms, r_max=5.0):
    """
    Converts an ASE Atoms object into a PyG Data object ready for E3GNN.
    
    Args:
        atoms (ase.Atoms): The structural data with PBC and tags.
        r_max (float): Neighbor cutoff radius.
        
    Returns:
        torch_geometric.data.Data: Prepared graph.
    """
    # 1. Structural tensors
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float)  # (3, 3)

    # 2. PBC-aware neighbour list
    idx_i, idx_j, S, D = neighbor_list("ijSD", atoms, cutoff=r_max)

    edge_index = torch.tensor(np.stack([idx_i, idx_j]), dtype=torch.long)  # (2, E)
    edge_shift = torch.tensor(S @ np.array(atoms.get_cell()), dtype=torch.float)  # (E, 3)

    # 3. Absorber sites
    tags = atoms.get_tags()
    absorber_indices = np.where(tags == 1)[0]

    if len(absorber_indices) == 0:
        raise ValueError("No absorber tag (tag=1) found in structure.")

    absorber_mask = torch.zeros(len(z), dtype=torch.bool)
    absorber_mask[absorber_indices] = True

    data = Data(
        z=z,
        pos=pos,
        cell=cell,
        edge_index=edge_index,
        edge_shift=edge_shift,
        absorber_mask=absorber_mask,
    )
    return data

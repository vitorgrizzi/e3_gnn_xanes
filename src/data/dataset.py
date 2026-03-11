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
        processed_path: str | None = None,
        r_max: float = 5.0,
        emin: float = -30.0,
        emax: float = 100.0,
        num_energy_points: int = 150,
        preprocess: bool = False,
        filter_multiple_absorbers: bool = False,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory where processed PyG files are cached.
            db_path: Path to the ASE SQLite database file.
            processed_path: Optional direct path to a .pt file. Overrides root/db logic if provided.
            r_max: Cutoff radius (Å) for graph connectivity.
            emin / emax / num_energy_points: Target energy grid for spectrum interpolation.
            preprocess: If True, forces reprocessing of the dataset even if processed files exist.
            filter_multiple_absorbers: If True, only keep graphs with exactly one absorber atom.
        """
        self.db_path = db_path
        self.r_max = r_max
        self.emin = emin
        self.emax = emax
        self.num_energy_points = num_energy_points
        self.target_energy_grid = torch.linspace(emin, emax, num_energy_points)
        self.filter_multiple_absorbers = filter_multiple_absorbers
        
        # 1. Direct path loading
        if processed_path is not None and os.path.exists(processed_path):
            super().__init__(root, transform, pre_transform)
            self.load(processed_path)
            return

        # 2. Force re-processing if requested
        processed_file = self.processed_file_names[0]
        full_cache_path = os.path.join(root, "processed", processed_file)
        if preprocess and os.path.exists(full_cache_path):
            print(f"Preprocess=True: Deleting cache to force rebuild: {processed_file}")
            os.remove(full_cache_path)

        # 3. Super init triggers process() automatically if files are missing
        super().__init__(root, transform, pre_transform)
        # 'pre_transform' is applied only when the graph is created during process(); changes are saved to disk.
        # 'transform' is applied to each graph after it is loaded from the cache; changes are not saved to disk.

        # 4. Load processed data into memory
        if os.path.exists(self.processed_paths[0]):
            self.load(self.processed_paths[0])
        # Note that processed_path is created by the super().__init__() method

    @staticmethod
    def normalize_xanes(E=None, mu=None, E0=None, pre_range=(-55, -20), post_range=(80, 150), post_order=1):
        """
        Normalize a XANES spectrum using pre-edge subtraction and post-edge scaling.
        """
        if E is not None and mu is not None:
            E = np.asarray(E, dtype=float)
            mu = np.asarray(mu, dtype=float)
        else:
            raise ValueError("Provide both E and mu.")

        if E0 is None:
            if np.min(E) <= 0 <= np.max(E):
                E0 = 0.0
            else:
                dmu_dE = np.gradient(mu, E)
                E0 = E[np.argmax(dmu_dE)]

        # Define fitting windows
        pre_mask = (E >= E0 + pre_range[0]) & (E <= E0 + pre_range[1])
        post_mask = (E >= E0 + post_range[0]) & (E <= E0 + post_range[1])

        if pre_mask.sum() < 2 or post_mask.sum() < post_order + 1:
            raise ValueError("Not enough points in pre-edge or post-edge ranges for normalization.")

        # Fits
        p_pre = np.polyfit(E[pre_mask], mu[pre_mask], 1)
        p_post = np.polyfit(E[post_mask], mu[post_mask], post_order)

        pre_line = np.polyval(p_pre, E)
        step = np.polyval(p_post, E0) - np.polyval(p_pre, E0)

        if np.isclose(step, 0.0):
            raise ValueError("Computed edge step is zero or too close to zero.")

        mu_norm = (mu - pre_line) / step
        return np.column_stack([E, mu_norm]), np.column_stack([E, mu])

    # ------------------------------------------------------------------
    # PyG boilerplate
    # ------------------------------------------------------------------
    @property
    def processed_file_names(self):
        return [f"data_rmax{self.r_max}_e{self.num_energy_points}.pt"]

    # ------------------------------------------------------------------
    # Processing the database and turning it into a graph
    # ------------------------------------------------------------------
    # Note that process() is called automatically by the super().__init__() method
    # if the processed files do not exist. It is a magic method present in the super 
    # class that PyG expect us to override.
    def process(self):
        if self.db_path is None:
            raise RuntimeError(
                "PyG is attempting to process the dataset (because the processed .pt cache "
                "file was not found), but `db_path` is None. Please provide `data.db_path` "
                "in your config or CLI arguments so the dataset can be built."
            )

        data_list = []
        uniform_energy_grid = self.target_energy_grid.numpy()

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
                
                # Apply the proper physical normalization before interpolation since the 
                # energy range of the raw data is larger than `uniform_energy_grid`
                try:
                    norm_xas, _ = self.normalize_xanes(E=raw_e, mu=raw_y)
                    norm_y = norm_xas[:, 1]
                except Exception as e:
                    print(f"Skipping row id={row.id}: Normalization failed - {e}")
                    continue

                # Interpolating the xanes using `uniform_energy_grid`
                interp_y = np.interp(uniform_energy_grid, raw_e, norm_y)
                y = torch.tensor(interp_y, dtype=torch.float).unsqueeze(0)
                # y has (1, N_E) shape to indicate PyG that `y` is a graph property, not a node property.

                # Use helper to build graph
                try:
                    data = atoms_to_graph(atoms, r_max=self.r_max)
                    data.y = y # Add target
                    
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                        
                    if self.filter_multiple_absorbers and data.absorber_mask.sum() != 1:
                        continue

                    data_list.append(data)
                except ValueError as e:
                    print(f"Skipping row id={row.id}: {e}")
                    continue

        if len(data_list) > 0:
            print(f"Processed {len(data_list)} graphs from {self.db_path} with proper physical edge-step normalization.")
        self.save(data_list, self.processed_paths[0])


def atoms_to_graph(atoms, r_max=6.0):
    """
    Converts an ASE Atoms object into a PyG Data object ready for E3GNN.
    
    Args:
        atoms (ase.Atoms): The structural data with PBC and tags.
        r_max (float): Neighbor cutoff radius.
        
    Returns:
        torch_geometric.data.Data: Prepared graph.
    """
    # 1. Structural tensors
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long) # (N,)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float) # (N, 3)
    cell = torch.tensor(atoms.get_cell(), dtype=torch.float) # (3, 3)

    # 2. PBC-aware neighbour list
    idx_i, idx_j, S = neighbor_list("ijS", atoms, cutoff=r_max) # [(E,), (E,), (E, 3)]
    edge_index = torch.tensor(np.stack([idx_i, idx_j], axis=0), dtype=torch.long) # (2, E)
    edge_shift = torch.tensor(S @ cell.numpy(), dtype=torch.float) # (E, 3)
    # `S` is a matrix of integer PBC shifts for each edge. True PBC distance is thus 
    # D = r_j - r_i + S @ cell, where r_i and r_j are always the (x,y,z) of the atoms 
    # in the "real" unit cell.

    # 3. Absorber sites
    absorber_mask = torch.tensor(atoms.get_tags(), dtype=torch.bool)

    if not absorber_mask.any():
        raise ValueError("No absorber tag (tag=1) found in structure.")

    # 4. Create PyG Data object
    data = Data(
        edge_index=edge_index, 
        pos=pos,
        z=z, # kwargs from here on
        cell=cell,
        edge_shift=edge_shift,
        absorber_mask=absorber_mask,
    ) # Note that 'y' is added later in the process() method.

    return data

if __name__ == "__main__":
    # Example usage / Testing script
    import argparse
    
    parser = argparse.ArgumentParser(description="Test XANES dataset processing.")
    parser.add_argument("--db", type=str, default="xanes_data.db", help="Path to ASE SQLite database.")
    parser.add_argument("--root", type=str, default="data/processed_test", help="Root directory for PyG processing.")
    parser.add_argument("--rmax", type=float, default=5.0, help="Neighbor cutoff radius.")
    parser.add_argument("--filter-multiple-absorbers", default=False, help="Only keep graphs with exactly one absorber.")
    args = parser.parse_args()

    # Initialize dataset
    print(f"--- Initializing Dataset ---")
    print(f"DB Path: {args.db}")
    print(f"Processed Root: {args.root}")
    
    dataset = XANESDataset(
        root=args.root,
        db_path=args.db,
        r_max=args.rmax,
        filter_multiple_absorbers=args.filter_multiple_absorbers,
        preprocess=True # Force rebuild for testing
    )

    print(f"\n--- Dataset Summary ---")
    print(f"Number of graphs: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n--- Sample Graph Inspection ---")
        print(f"Data object: {sample}")
        print(f"z (Atomic numbers): {sample.z.shape} -> {sample.z[:5]}...")
        print(f"pos (Positions): {sample.pos.shape}")
        print(f"edge_index: {sample.edge_index.shape}")
        print(f"edge_shift: {sample.edge_shift.shape}")
        print(f"y (Spectrum): {sample.y.shape}")
        print(f"Absorber count: {sample.absorber_mask.sum().item()}")

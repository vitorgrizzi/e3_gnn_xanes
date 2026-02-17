"""
Self-contained test for the rewritten XANESDataset.

Creates a temporary ASE DB with synthetic periodic structures,
instantiates XANESDataset, and validates the resulting Data objects.
"""
import os
import shutil
import tempfile

import numpy as np
import torch
from ase import Atoms
from ase.db import connect

from src.data.dataset import XANESDataset


def _make_test_db(db_path: str):
    """Create a tiny ASE DB with two structures."""
    with connect(db_path, append=False) as db:
        # --- Structure 1: simple cubic Fe with 1 absorber ---
        atoms1 = Atoms(
            "Fe4",
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        )
        tags1 = np.array([1, 0, 0, 0])
        atoms1.set_tags(tags1)

        e1 = np.linspace(-30, 100, 200)
        y1 = np.exp(-((e1 - 10) ** 2) / 50)
        spectrum1 = np.column_stack([e1, y1])

        db.write(atoms1, data={"xanes": spectrum1})

        # --- Structure 2: Fe2O3-like with 3 absorbers ---
        atoms2 = Atoms(
            "Fe3O5",
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
                [0.1, 0.2, 0.3],
                [0.4, 0.1, 0.6],
                [0.7, 0.8, 0.2],
                [0.3, 0.6, 0.9],
                [0.9, 0.4, 0.1],
            ],
            cell=[5.0, 5.2, 5.4],
            pbc=True,
        )
        tags2 = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        atoms2.set_tags(tags2)

        e2 = np.linspace(-30, 100, 300)
        y2 = np.exp(-((e2 - 20) ** 2) / 80)
        spectrum2 = np.column_stack([e2, y2])

        db.write(atoms2, data={"xanes": spectrum2})


def test_xanes_dataset():
    tmp = tempfile.mkdtemp(prefix="xanes_test_")
    db_path = os.path.join(tmp, "test.db")
    root = os.path.join(tmp, "dataset_root")

    try:
        _make_test_db(db_path)

        r_max = 5.0
        num_e = 150

        dataset = XANESDataset(
            root=root,
            db_path=db_path,
            r_max=r_max,
            emin=-30.0,
            emax=100.0,
            num_energy_points=num_e,
        )

        # --- Count: 1 absorber + 3 absorbers = 4 graphs ---
        assert len(dataset) == 4, f"Expected 4, got {len(dataset)}"

        for i, data in enumerate(dataset):
            # Required fields
            assert hasattr(data, "z"), f"Graph {i}: missing z"
            assert hasattr(data, "pos"), f"Graph {i}: missing pos"
            assert hasattr(data, "cell"), f"Graph {i}: missing cell"
            assert hasattr(data, "edge_index"), f"Graph {i}: missing edge_index"
            assert hasattr(data, "edge_shift"), f"Graph {i}: missing edge_shift"
            assert hasattr(data, "absorber_mask"), f"Graph {i}: missing absorber_mask"
            assert hasattr(data, "y"), f"Graph {i}: missing y"

            # Shapes
            n = data.z.size(0)
            e = data.edge_index.size(1)
            assert data.pos.shape == (n, 3), f"Graph {i}: bad pos shape"
            assert data.cell.shape == (3, 3), f"Graph {i}: bad cell shape"
            assert data.edge_shift.shape == (e, 3), f"Graph {i}: bad edge_shift shape"
            assert data.absorber_mask.shape == (n,), f"Graph {i}: bad mask shape"
            assert data.y.shape == (1, num_e), f"Graph {i}: bad y shape {data.y.shape}"

            # Exactly one absorber per graph
            assert data.absorber_mask.sum().item() == 1, (
                f"Graph {i}: expected 1 absorber, got {data.absorber_mask.sum()}"
            )

            # Edge vectors within cutoff
            src, dst = data.edge_index
            vec = data.pos[dst] - data.pos[src] + data.edge_shift
            norms = vec.norm(dim=1)
            assert norms.max().item() <= r_max + 1e-4, (
                f"Graph {i}: max edge length {norms.max():.4f} > r_max={r_max}"
            )

        print("✅ All assertions passed!")
        print(f"   Graphs: {len(dataset)}")
        for i, d in enumerate(dataset):
            abs_z = d.z[d.absorber_mask].item()
            print(
                f"   [{i}] atoms={d.z.size(0)}, edges={d.edge_index.size(1)}, "
                f"absorber_Z={abs_z}, y_shape={tuple(d.y.shape)}, "
                f"cell_diag={d.cell.diag().tolist()}"
            )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_xanes_dataset()

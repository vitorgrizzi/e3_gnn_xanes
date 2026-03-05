import torch

from src.model import MultiScaleGaussianBasis


def test_basis_shape_and_positive_sigmas():
    basis = MultiScaleGaussianBasis(
        n_basis=24,
        emin=-30.0,
        emax=100.0,
        scales_ratios=[0.5, 1.0, 2.0],
        global_bg=True,
    )
    energy_grid = torch.linspace(-30.0, 100.0, 150)

    matrix = basis(energy_grid)

    assert matrix.shape == (150, 25)
    assert torch.all(basis.sigmas.detach() > 0)


def test_basis_progressively_flattens_with_scale():
    basis = MultiScaleGaussianBasis(
        n_basis=24,
        emin=-30.0,
        emax=100.0,
        scales_ratios=[0.5, 1.0, 2.0],
        global_bg=False,
        focus_energy=15.0,
        min_uniform_weight=0.0,
        max_uniform_weight=0.95,
    )

    grouped_centers = [basis.centers[s] for s in basis.scale_slices]
    focus = basis.focus_energy

    mean_abs_distance = [torch.mean(torch.abs(group - focus)).item() for group in grouped_centers]
    spacing_cv = []
    for group in grouped_centers:
        diffs = torch.diff(group)
        spacing_cv.append((diffs.std() / diffs.mean()).item())

    assert mean_abs_distance[0] < mean_abs_distance[-1]
    assert spacing_cv[0] > spacing_cv[-1]


def test_focus_energy_moves_the_sharpest_scale():
    left_focused = MultiScaleGaussianBasis(
        n_basis=18,
        emin=-30.0,
        emax=100.0,
        scales_ratios=[0.5, 2.0],
        global_bg=False,
        focus_energy=5.0,
    )
    right_focused = MultiScaleGaussianBasis(
        n_basis=18,
        emin=-30.0,
        emax=100.0,
        scales_ratios=[0.5, 2.0],
        global_bg=False,
        focus_energy=35.0,
    )

    sharp_slice = left_focused.scale_slices[0]
    assert right_focused.centers[sharp_slice].mean() > left_focused.centers[sharp_slice].mean()
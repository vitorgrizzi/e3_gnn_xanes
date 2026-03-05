import torch
import torch.nn as nn


class MultiScaleGaussianBasis(nn.Module):
    """
    Multi-scale Gaussian basis for spectral reconstruction.

    The sharpest scale is concentrated around ``focus_energy`` using a right-skewed
    density, and broader scales progressively flatten toward a uniform placement.
    """

    def __init__(
        self,
        n_basis=128,
        emin=-10.0,
        emax=50.0,
        scales_ratios=[0.1, 0.5, 1.0],
        global_bg=True,
        focus_energy=15.0,
        focus_left_width_ratio=0.05,
        focus_right_width_ratio=0.25,
        min_uniform_weight=0.0,
        max_uniform_weight=0.9,
        flatten_exponent=1.0,
        cdf_resolution=4096,
        peak_e=None,
    ):
        """
        Args:
            n_basis (int): Total number of basis functions.
            emin (float): Minimum energy value.
            emax (float): Maximum energy value.
            scales_ratios (list): Relative widths for each Gaussian group.
            focus_energy (float): Energy around which the sharpest scale is concentrated.
            focus_left_width_ratio (float): Relative width of the left side of the skewed density.
            focus_right_width_ratio (float): Relative width of the right side of the skewed density.
            min_uniform_weight (float): Uniform mixing weight for the sharpest scale.
            max_uniform_weight (float): Uniform mixing weight for the broadest scale.
            flatten_exponent (float): Controls how quickly the distribution flattens with scale.
            cdf_resolution (int): Number of points used for numerical CDF inversion.
            peak_e (float | None): Backward-compatible alias for ``focus_energy``.
        """
        super().__init__()

        if peak_e is not None:
            focus_energy = peak_e

        if n_basis <= 0:
            raise ValueError("n_basis must be positive.")
        if len(scales_ratios) == 0:
            raise ValueError("scales_ratios must contain at least one scale.")
        if any(r <= 0 for r in scales_ratios):
            raise ValueError("scales_ratios must be strictly positive.")
        if emax <= emin:
            raise ValueError("emax must be greater than emin.")
        if focus_left_width_ratio <= 0 or focus_right_width_ratio <= 0:
            raise ValueError("Focus width ratios must be positive.")
        if cdf_resolution < 2:
            raise ValueError("cdf_resolution must be at least 2.")

        self.n_basis = n_basis
        self.emin = emin
        self.emax = emax
        self.global_bg = global_bg
        self.focus_energy = focus_energy
        self.focus_left_width_ratio = focus_left_width_ratio
        self.focus_right_width_ratio = focus_right_width_ratio
        self.min_uniform_weight = min_uniform_weight
        self.max_uniform_weight = max_uniform_weight
        self.flatten_exponent = flatten_exponent
        self.cdf_resolution = cdf_resolution

        if self.global_bg:
            self.bg_center = nn.Parameter(torch.tensor(0.0))
            self.bg_width = nn.Parameter(torch.tensor(2.0))

        n_scales = len(scales_ratios)
        n_per_scale = n_basis // n_scales
        remainder = n_basis % n_scales

        centers_list = []
        sigmas_list = []
        scale_slices = []
        scale_uniform_weights = []

        full_range = float(emax - emin)
        fine_grid = torch.linspace(emin, emax, cdf_resolution)
        skewed_pdf = self._build_skewed_pdf(
            fine_grid=fine_grid,
            focus_energy=focus_energy,
            full_range=full_range,
            left_width_ratio=focus_left_width_ratio,
            right_width_ratio=focus_right_width_ratio,
        )
        uniform_pdf = torch.full_like(fine_grid, 1.0 / fine_grid.numel())
        min_ratio = min(scales_ratios)
        max_ratio = max(scales_ratios)

        offset = 0
        for i, ratio in enumerate(scales_ratios):
            count = n_per_scale + (1 if i < remainder else 0)
            scale_slices.append(slice(offset, offset + count))
            offset += count

            uniform_weight = self._uniform_weight(
                ratio=ratio,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                min_uniform_weight=min_uniform_weight,
                max_uniform_weight=max_uniform_weight,
                flatten_exponent=flatten_exponent,
            )
            scale_uniform_weights.append(uniform_weight)
            blended_pdf = (1.0 - uniform_weight) * skewed_pdf + uniform_weight * uniform_pdf

            cdf = torch.cumsum(blended_pdf, dim=0)
            cdf = cdf / cdf[-1]
            target_probs = torch.linspace(0, 1, count)
            centers = self._invert_cdf(fine_grid, cdf, target_probs)

            local_spacing = self._local_spacing(centers, full_range)
            sigma = local_spacing * ratio * 2.0

            centers_list.append(centers)
            sigmas_list.append(sigma)

        self.scale_counts = [s.stop - s.start for s in scale_slices]
        self.scale_slices = scale_slices
        self.scale_uniform_weights = scale_uniform_weights

        self.register_buffer("centers", torch.cat(centers_list))
        self.sigmas = nn.Parameter(torch.cat(sigmas_list))

    def forward(self, energy_grid):
        """
        Evaluate the basis functions on the provided energy grid.

        Args:
            energy_grid (torch.Tensor): Shape [N_E], energy points.

        Returns:
            torch.Tensor: Basis matrix B of shape [N_E, n_basis].
        """
        x_mu_diff = energy_grid.unsqueeze(1) - self.centers.unsqueeze(0)
        sigmas = self.sigmas.unsqueeze(0).clamp(min=1e-4)
        B = torch.exp(-(x_mu_diff ** 2) / (2 * sigmas ** 2))

        if self.global_bg:
            bg_width = self.bg_width.clamp(min=1e-3)
            bg = torch.sigmoid((energy_grid - self.bg_center) / bg_width)
            B = torch.cat([B, bg.unsqueeze(1)], dim=1)

        return B

    @staticmethod
    def _build_skewed_pdf(fine_grid, focus_energy, full_range, left_width_ratio, right_width_ratio):
        focus = float(min(max(focus_energy, fine_grid[0].item()), fine_grid[-1].item()))
        left_width = full_range * left_width_ratio + 1e-6
        right_width = full_range * right_width_ratio + 1e-6
        left_pdf = torch.exp(-0.5 * ((fine_grid - focus) / left_width) ** 2)
        right_pdf = torch.exp(-0.5 * ((fine_grid - focus) / right_width) ** 2)
        skewed_pdf = torch.where(fine_grid <= focus, left_pdf, right_pdf)
        return skewed_pdf / skewed_pdf.sum()

    @staticmethod
    def _uniform_weight(
        ratio,
        min_ratio,
        max_ratio,
        min_uniform_weight,
        max_uniform_weight,
        flatten_exponent,
    ):
        if max_ratio == min_ratio:
            progress = 1.0
        else:
            progress = (ratio - min_ratio) / (max_ratio - min_ratio)
        progress = float(progress) ** float(flatten_exponent)
        return float(min_uniform_weight + progress * (max_uniform_weight - min_uniform_weight))

    @staticmethod
    def _invert_cdf(fine_grid, cdf, target_probs):
        if fine_grid.numel() == 1:
            return fine_grid.expand_as(target_probs)

        idx = torch.searchsorted(cdf, target_probs, right=False)
        idx = idx.clamp(min=1, max=fine_grid.numel() - 1)

        cdf_lo = cdf[idx - 1]
        cdf_hi = cdf[idx]
        grid_lo = fine_grid[idx - 1]
        grid_hi = fine_grid[idx]

        denom = (cdf_hi - cdf_lo).clamp(min=1e-12)
        weight = (target_probs - cdf_lo) / denom
        return grid_lo + weight * (grid_hi - grid_lo)

    @staticmethod
    def _local_spacing(centers, full_range):
        if centers.numel() == 1:
            return torch.tensor([full_range], dtype=centers.dtype, device=centers.device)

        local_spacing = torch.empty_like(centers)
        local_spacing[0] = centers[1] - centers[0]
        local_spacing[-1] = centers[-1] - centers[-2]
        if centers.numel() > 2:
            local_spacing[1:-1] = (centers[2:] - centers[:-2]) / 2.0
        return local_spacing.clamp(min=1e-4)

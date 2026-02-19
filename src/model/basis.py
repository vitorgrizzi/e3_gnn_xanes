import torch
import torch.nn as nn
import numpy as np

class MultiScaleGaussianBasis(nn.Module):
    """
    Multi-Scale Gaussian Basis for spectral reconstruction.
    
    This module generates a basis of Gaussian functions with varying widths (scales)
    to capture both sharp features (like the white line) and broad oscillations.

    OBS1: We are defining three possible widths for the Gaussian via the `scales_ratios` 
          parameter instead of allowing the model to learn the widths. This is done to 
          reduce the number of parameters and make the loss landscape smoother (easier to train).
    """
    def __init__(self, n_basis=128, emin=-10.0, emax=50.0, scales_ratios=[0.1, 0.5, 1.0]):
        """
        Args:
            n_basis (int): Total number of basis functions.
            emin (float): Minimum energy value.
            emax (float): Maximum energy value.
            scales_ratios (list): Relative widths of the Gaussians for the 3 scales. 
                                  Will be scaled by the grid spacing.
        """
        super().__init__()
        self.n_basis = n_basis
        self.emin = emin
        self.emax = emax
        
        # We split the basis roughly equally among the scales
        n_scales = len(scales_ratios)
        n_per_scale = n_basis // n_scales
        remainder = n_basis % n_scales
        
        centers_list = []
        sigmas_list = []
        
        # Base grid spacing
        full_range = emax - emin
        
        for i, ratio in enumerate(scales_ratios):
            count = n_per_scale + (1 if i < remainder else 0)
            
            # Distribute centers evenly
            # We add a small buffer so centers aren't exactly on the edge if we don't want
            c = torch.linspace(emin, emax, count)
            
            # Width is proportional to spacing, modified by ratio
            # Or simplified: specific width per scale
            # Let's use the user's logic: distinct widths.
            # We estimate a "standard" width as range / count
            spacing = full_range / count
            sigma = spacing * ratio * 2.0 # Factor of 2 for overlap
            
            centers_list.append(c)
            sigmas_list.append(torch.full_like(c, sigma))
            
        self.register_buffer('centers', torch.cat(centers_list))
        self.register_buffer('sigmas', torch.cat(sigmas_list))
        
    def forward(self, energy_grid):
        """
        Evaluate the basis functions on the provided energy grid.
        
        Args:
            energy_grid (torch.Tensor): Shape [N_E], energy points.
            
        Returns:
            torch.Tensor: Basis matrix B of shape [N_E, n_basis].
                          B[i, j] = exp(- (E_i - mu_j)^2 / (2*sigma_j^2))
        """
        # [N_E, 1] - [1, n_basis] -> [N_E, n_basis]
        diff = energy_grid.unsqueeze(1) - self.centers.unsqueeze(0)
        B = torch.exp(-(diff**2) / (2 * self.sigmas.unsqueeze(0)**2))
        return B

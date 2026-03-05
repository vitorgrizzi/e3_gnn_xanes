import torch
import torch.nn as nn

class MultiScaleGaussianBasis(nn.Module):
    """
    Multi-Scale Gaussian Basis for spectral reconstruction.
    
    This module generates a basis of Gaussian functions with fixed widths (scales)
    and learnable centers to capture both sharp features and broad oscillations.

    OBS1: We are fixing the Gaussian widths and learning their centers.
    """
    def __init__(self, n_basis=128, emin=-10.0, emax=50.0, scales_ratios=[0.1, 0.5, 1.0], global_bg=True):
        """
        Args:
            n_basis (int): Total number of basis functions.
            emin (float): Minimum energy value.
            emax (float): Maximum energy value.
            scales_ratios (list): Relative initial widths/scales for each Gaussian group. There 
                                  will be `len(scales_ratios)` Gaussian groups.
        """
        super().__init__()
        self.n_basis = n_basis
        self.emin = emin
        self.emax = emax
        self.global_bg = global_bg
        
        if self.global_bg:
            # We add a learnable logistic/sigmoid function to act as an edge step background
            # that saturates at energies > 0 and stays low at energies < 0
            self.bg_center = nn.Parameter(torch.tensor(0.0))
            self.bg_width = nn.Parameter(torch.tensor(2.0)) # Initial soft step

        
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
            
            # Distribute centers evenly across the energy range
            centers = torch.linspace(emin, emax, count)
            
            # Finding the spacing between Gaussians
            spacing = full_range / count

            # Computing constant `sigma` for Gaussian group
            sigma = spacing * ratio * 2.0 # Factor of 2 for overlap
            
            # If remainder == 0, all Gaussian groups are initialized at the same point in the 
            # grid. During learning, the different centers will be adjusted to better fit the data
            centers_list.append(centers)
            sigmas_list.append(torch.full_like(centers, sigma))
            
        # Keep centers learnable so the network can adjust peak locations
        self.centers = nn.Parameter(torch.cat(centers_list))
        
        # register_buffer to store constants the model needs but are not learnable parameters, we can 
        # access it via self.sigmas in the forward pass
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
        x_mu_diff = energy_grid.unsqueeze(1) - self.centers.unsqueeze(0)

        # Avoid clamping sigmas as they are fixed, but applying clamping just in case
        sigmas = self.sigmas.unsqueeze(0).clamp(min=1e-4) # [1, n_basis]

        B = torch.exp(-(x_mu_diff**2) / (2*sigmas**2)) # [N_E, n_basis]
        
        if self.global_bg:
            # Clamp width to prevent division by zero or overly sharp step that causes NaNs
            bg_width = self.bg_width.clamp(min=1e-3)
            # Sigmoid models the cumulative edge step background
            bg = torch.sigmoid((energy_grid - self.bg_center) / bg_width) # [N_E]
            B = torch.cat([B, bg.unsqueeze(1)], dim=1) # [N_E, n_basis + 1]
            
        return B

import torch
import torch.nn as nn

class MultiScaleGaussianBasis(nn.Module):
    """
    Multi-Scale Gaussian Basis for spectral reconstruction.
    
    This module generates a basis of Gaussian functions with learnable widths (scales)
    and fixed centers to capture both sharp features and broad oscillations.

    OBS1: We are fixing the Gaussian centers and learning their widths. This is done to 
          reduce the number of parameters and make the loss landscape smoother (easier to train).
    """
    def __init__(self, n_basis=128, emin=-10.0, emax=50.0, scales_ratios=[0.1, 0.5, 1.0], global_bg=True, peak_e=15.0):
        """
        Args:
            n_basis (int): Total number of basis functions.
            emin (float): Minimum energy value.
            emax (float): Maximum energy value.
            scales_ratios (list): Relative initial widths/scales for each Gaussian group. There 
                                  will be `len(scales_ratios)` Gaussian groups.
            peak_e (float): The energy value where XANES transitions are most dense, used
                            to center the right-skewed Gaussian center distribution.
        """
        super().__init__()
        self.n_basis = n_basis
        self.emin = emin
        self.emax = emax
        self.global_bg = global_bg
        self.peak_e = peak_e
        
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
        max_ratio = max(scales_ratios)
        
        # Create a numerical CDF for the right-skewed peak density
        fine_grid = torch.linspace(emin, emax, 5000)
        w_left = full_range * 0.05 + 1e-6
        w_right = full_range * 0.25 + 1e-6
        
        skewed_pdf = torch.where(fine_grid < peak_e,
                                 torch.exp(-0.5 * ((fine_grid - peak_e) / w_left)**2),
                                 torch.exp(-0.5 * ((fine_grid - peak_e) / w_right)**2))
        skewed_pdf = skewed_pdf / skewed_pdf.sum()
        
        uniform_pdf = torch.ones_like(fine_grid) / len(fine_grid)
        
        for i, ratio in enumerate(scales_ratios):
            count = n_per_scale + (1 if i < remainder else 0)
            
            # Blend between skewed PDF and uniform PDF based on scale ratio. 
            # Sharpest scale (lowest ratio) is highly skewed. Largest scale is nearly uniform.
            uniform_weight = ratio / max_ratio
            blended_pdf = (1.0 - uniform_weight) * skewed_pdf + uniform_weight * uniform_pdf
            
            # Compute numerical CDF
            cdf = torch.cumsum(blended_pdf, dim=0)
            cdf = cdf / cdf[-1]
            
            # Map uniform percentiles back into energy grid
            target_probs = torch.linspace(0, 1, count)
            indices = torch.searchsorted(cdf, target_probs)
            indices = torch.clamp(indices, 0, len(fine_grid) - 1)
            centers = fine_grid[indices]
            
            # Initial `sigma`: estimate local spacing between points so denser regions 
            # get sharper initial Gaussians, while tails remain appropriately broader
            if count > 1:
                local_spacing = torch.zeros(count)
                local_spacing[0] = centers[1] - centers[0]
                local_spacing[-1] = centers[-1] - centers[-2]
                if count > 2:
                    local_spacing[1:-1] = (centers[2:] - centers[:-2]) / 2.0
            else:
                local_spacing = torch.tensor([full_range])
                
            sigma = local_spacing * ratio * 2.0 
            
            centers_list.append(centers)
            sigmas_list.append(sigma)
            
        # Freeze centers so they don't abandon the tail regions (prevents downward slope at boundaries) 
        self.register_buffer('centers', torch.cat(centers_list)) 
        # register_buffer to store constants the model needs but are not learnable parameters, we can 
        # access it via self.centers in the forward pass

        # Keep sigmas learnable so the network can adjust peak sharpness
        self.sigmas = nn.Parameter(torch.cat(sigmas_list))
        
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

        # Clamp sigmas to prevent division by zero in case the optimizer pushes them too small
        sigmas = self.sigmas.unsqueeze(0).clamp(min=1e-4) # [1, n_basis]

        B = torch.exp(-(x_mu_diff**2) / (2*sigmas**2)) # [N_E, n_basis]
        
        if self.global_bg:
            # Clamp width to prevent division by zero or overly sharp step that causes NaNs
            bg_width = self.bg_width.clamp(min=1e-3)
            # Sigmoid models the cumulative edge step background
            bg = torch.sigmoid((energy_grid - self.bg_center) / bg_width) # [N_E]
            B = torch.cat([B, bg.unsqueeze(1)], dim=1) # [N_E, n_basis + 1]
            
        return B

import torch
import torch.nn as nn

class SpectrumLoss(nn.Module):
    """
    Combined loss for XANES spectra:
    L = L_MSE + lambda * L_Gradient
    
    Ensures both intensity values and spectral shape (derivatives) are matched.
    """
    def __init__(self, lambda_grad=0.5):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.mse = nn.MSELoss()
        
    def forward(self, pred_y, true_y, energy_grid=None):
        """
        Args:
            pred_y: [Batch, N_E]
            true_y: [Batch, N_E]
            energy_grid: [N_E] (optional, needed if grid is non-uniform for gradients)
        """
        # 1. Intensity MSE
        loss_0 = self.mse(pred_y, true_y)
        
        # 2. Gradient MSE
        # Compute numerical gradient along energy axis (dim 1)
        # diff[i] = y[i+1] - y[i]
        # We can just use simple finite diff
        diff_pred = pred_y[:, 1:] - pred_y[:, :-1]
        diff_true = true_y[:, 1:] - true_y[:, :-1]
        
        loss_1 = self.mse(diff_pred, diff_true)
        
        total_loss = loss_0 + self.lambda_grad * loss_1
        return total_loss, loss_0, loss_1

import torch
import torch.nn as nn

class SpectrumLoss(nn.Module):
    """
    Combined loss for XANES spectra:
    L = L_MSE + lambda_1 * L_Gradient + lambda_2 * L_Laplacian
    
    Ensures both intensity values and spectral shape (derivatives) are matched.
    """
    def __init__(self, lambda_grad=0.5, lambda_lap=0.0):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_lap = lambda_lap
        self.mse = nn.MSELoss(reduction='mean')
        
    def forward(self, pred_y, true_y, energy_grid=None):
        """
        Args:
            pred_y: [batch_size, N_E]
            true_y: [batch_size, N_E]
        """
        # 1. Intensity MSE
        loss_0 = self.mse(pred_y, true_y)
        
        # 2. Gradient MSE (1st derivative)
        diff_pred = pred_y[:, 1:] - pred_y[:, :-1]
        diff_true = true_y[:, 1:] - true_y[:, :-1]
        loss_1 = self.mse(diff_pred, diff_true) 

        # 3. Laplacian MSE (2nd derivative)
        loss_2 = torch.tensor(0.0, device=pred_y.device)
        if self.lambda_lap > 0:
            lap_pred = diff_pred[:, 1:] - diff_pred[:, :-1]
            lap_true = diff_true[:, 1:] - diff_true[:, :-1]
            loss_2 = self.mse(lap_pred, lap_true)
        
        total_loss = loss_0 + (self.lambda_grad * loss_1) + (self.lambda_lap * loss_2) 
        return total_loss, loss_0, loss_1, loss_2

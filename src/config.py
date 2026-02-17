"""
Configuration dataclasses for model and training hyperparameters.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class ModelConfig:
    """Configuration for XANES_E3GNN architecture."""
    max_z: int = 100
    num_layers: int = 4
    lmax: int = 2
    mul_0: int = 64
    mul_1: int = 32
    mul_2: int = 16
    r_max: float = 5.0
    num_basis: int = 128
    num_radial: int = 10
    basis_scales: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    emin: float = -10.0
    emax: float = 50.0


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    lr: float = 1e-3
    batch_size: int = 4
    epochs: int = 100
    lambda_grad: float = 0.5
    patience: int = 15
    grad_clip: float = 5.0
    save_path: Optional[str] = "best_model.pt"
    device: Optional[str] = None  # auto-detect if None

    def get_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_from_configs(model_cfg: ModelConfig, train_cfg: TrainingConfig):
    """
    Convenience factory: build model, criterion, and legacy config dict
    from typed config objects.
    
    Returns:
        (model, criterion, config_dict)
    """
    from src.model import XANES_E3GNN
    from src.loss import SpectrumLoss

    model = XANES_E3GNN(
        max_z=model_cfg.max_z,
        num_layers=model_cfg.num_layers,
        lmax=model_cfg.lmax,
        mul_0=model_cfg.mul_0,
        mul_1=model_cfg.mul_1,
        mul_2=model_cfg.mul_2,
        r_max=model_cfg.r_max,
        num_basis=model_cfg.num_basis,
        num_radial=model_cfg.num_radial,
        basis_scales=model_cfg.basis_scales,
        emin=model_cfg.emin,
        emax=model_cfg.emax,
    )

    criterion = SpectrumLoss(lambda_grad=train_cfg.lambda_grad)
    energy_grid = torch.linspace(model_cfg.emin, model_cfg.emax, 100)

    config = {
        'lr': train_cfg.lr,
        'batch_size': train_cfg.batch_size,
        'epochs': train_cfg.epochs,
        'criterion': criterion,
        'energy_grid': energy_grid,
        'save_path': train_cfg.save_path,
        'device': train_cfg.get_device(),
        'patience': train_cfg.patience,
        'grad_clip': train_cfg.grad_clip,
    }

    return model, criterion, config

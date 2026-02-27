import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def plot_spectra_comparison(
    energies, 
    spectrum_true, 
    spectrum_pred, 
    title=None, 
    ax=None, 
    show_legend=True
):
    """
    Plots a single comparison between true and predicted spectra.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(energies, spectrum_true, 'k-', lw=1.5, label='True', alpha=0.7)
    ax.plot(energies, spectrum_pred, 'r--', lw=1.2, label='Predicted')
    ax.fill_between(energies, spectrum_true, spectrum_pred, color='red', alpha=0.1)
    
    if title:
        ax.set_title(title, fontsize=10)
    
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend()
        
    return ax

def generate_validation_plots(
    model, 
    val_dataset, 
    energy_grid, 
    num_samples=6, 
    output_path="val_plots.png",
    device="cpu"
):
    """
    Selects random samples from validation dataset, runs inference, and saves a grid plot.
    """
    if len(val_dataset) == 0:
        return
        
    n = min(num_samples, len(val_dataset))
    indices = random.sample(range(len(val_dataset)), n)
    
    model.eval()
    
    # Calculate grid size
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), constrained_layout=True)
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    energies = energy_grid.cpu().numpy()

    for i in range(n):
        idx = indices[i]
        data = val_dataset[idx]
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        data = data.to(device)
        
        with torch.no_grad():
            spectrum_pred = model.predict_spectra(data, energy_grid).cpu().squeeze().numpy()
            spectrum_true = data.y.cpu().squeeze().numpy()

        mse = np.mean((spectrum_true - spectrum_pred)**2)
        
        plot_spectra_comparison(
            energies, 
            spectrum_true, 
            spectrum_pred, 
            title=f"Val Index: {idx}\nMSE: {mse:.4f}",
            ax=axes[i],
            show_legend=(i == 0)
        )
        
        if i % cols == 0: axes[i].set_ylabel("Intensity")
        if i >= (rows-1) * cols: axes[i].set_xlabel("Energy (eV)")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Validation plots saved to {output_path}")

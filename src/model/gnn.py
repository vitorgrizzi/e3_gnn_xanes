import torch
import torch.nn as nn
from e3nn import o3
# from e3nn.nn.models.gate_points_2102 import InteractionBlock # REMOVED
from torch_scatter import scatter # Will check if this works, else fallback handled in valid places?

from src.model.basis import MultiScaleGaussianBasis
from src.model.layers import AtomicEmbedding, CustomInteractionBlock
from src.model.pooling import AbsorberQueryAttention

class XANES_E3GNN(nn.Module):
    def __init__(self, 
                 max_z=100, 
                 num_layers=4, 
                 lmax=2, 
                 mul_0=64, 
                 mul_1=32, 
                 mul_2=16,
                 r_max=5.0,
                 num_basis=128,
                 basis_scales=[0.1, 0.5, 1.0],
                 emin=-10, emax=50):
        super().__init__()
        
        # 1. Embeddings & Irreps
        self.embedding = AtomicEmbedding(max_z, mul_0)
        
        # Define hidden irreps: e.g. 64x0e + 32x1o + 16x2e
        self.irreps_hidden = o3.Irreps(f"{mul_0}x0e + {mul_1}x1o + {mul_2}x2e")
        
        # Input irreps to first layer: just scalars (0e) from embedding
        self.irreps_in = o3.Irreps(f"{mul_0}x0e")
        
        # 2. Backbone Interaction Blocks
        self.layers = nn.ModuleList()
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        
        # Layer 0: scalars -> hidden
        self.layers.append(
            CustomInteractionBlock(
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_hidden,
                irreps_sh=self.irreps_sh,
                number_of_radial_basis_functions=10,
                steps=torch.linspace(0.0, r_max, 10),
            )
        )
        
        # Subsequent layers: hidden -> hidden
        for _ in range(num_layers - 1):
             self.layers.append(
                CustomInteractionBlock(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    number_of_radial_basis_functions=10,
                    steps=torch.linspace(0.0, r_max, 10),
                )
             )
             
        # 3. Pooling
        self.pooling = AbsorberQueryAttention(
            irep_in=self.irreps_hidden,
            hidden_dim=mul_0
        )
        
        # 4. Readout Head
        readout_dim = mul_0 + mul_0 + mul_1 + mul_2
        
        self.readout_mlp = nn.Sequential(
            nn.Linear(readout_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_basis)
        )
        
        # Basis
        self.basis = MultiScaleGaussianBasis(n_basis=num_basis, emin=emin, emax=emax, scales_ratios=basis_scales)
        
    def forward(self, data):
        """
        Args:
            data: PyG Data object
        """
        # Fallback for data.z vs data.x
        if hasattr(data, 'z') and data.z is not None:
             z = data.z
        else:
             z = data.x.squeeze().long()
             
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        
        # Embed
        h = self.embedding(z) # [N, mul_0]
        
        # Edge attributes
        edge_src, edge_dst = edge_index
        vec = pos[edge_dst] - pos[edge_src]
        edge_len = vec.norm(dim=1)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, vec, normalize=True, normalization='component')
        
        # Interaction Blocks
        for layer in self.layers:
            h = layer(x=h, edge_attr=edge_sh, edge_length=edge_len, edge_src=edge_src, edge_dst=edge_dst)
            
        # Extract Absorber State for Readout
        # h matches self.irreps_hidden
        # 64x0e + 32x1o + 16x2e
        # We need rigorous slicing based on irreps
        # e3nn might have evolved internal structures, but direct slicing is robust if layout is fixed.
        
        current_idx = 0
        slices = []
        for mul, ir in self.irreps_hidden:
            length = mul * ir.dim
            slices.append((current_idx, current_idx + length))
            current_idx += length
            
        # We assume order: 0e, 1o, 2e as defined in init string
        # slice 0: 64x0e
        # slice 1: 32x1o
        # slice 2: 16x2e
        
        s_range = slices[0]
        v_range = slices[1]
        t_range = slices[2]
        
        scalars_all = h[:, s_range[0]:s_range[1]]
        
        # Reshape vectors and tensors
        l1_all = h[:, v_range[0]:v_range[1]].reshape(-1, 32, 3)
        l2_all = h[:, t_range[0]:t_range[1]].reshape(-1, 16, 5)
        
        # Absorber specific features
        mask = data.absorber_mask
        s_a = scalars_all[mask] 
        v_a = l1_all[mask]      
        t_a = l2_all[mask]      
        
        # Norms
        norm_v = torch.sum(v_a**2, dim=-1) # [N_graph, 32]
        norm_t = torch.sum(t_a**2, dim=-1) # [N_graph, 16]
        
        # Context Pooling
        c = self.pooling(h, mask, batch) # [N_graph, 64]
        
        # Concatenate
        z_readout = torch.cat([s_a, c, norm_v, norm_t], dim=1) 
        
        # Predict Coefficients
        coeffs = self.readout_mlp(z_readout)
        
        return coeffs
    
    def predict_spectra(self, data, energy_grid):
        coeffs = self.forward(data) # [B, M]
        basis_matrix = self.basis(energy_grid) # [N_E, M]
        spectra = torch.matmul(basis_matrix, coeffs.T).T
        return spectra

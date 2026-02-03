import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn.models.gate_points_2102 import InteractionBlock
from e3nn.nn import FullyConnectedNet
from torch_scatter import scatter

from src.model.basis import MultiScaleGaussianBasis
from src.model.layers import AtomicEmbedding
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
        # Note: Parity depends on physics. 
        # Spherical harmonics: Y0: even, Y1: odd, Y2: even.
        # Let's align hidden features with that logic if we assume geometric nature.
        self.irreps_hidden = o3.Irreps(f"{mul_0}x0e + {mul_1}x1o + {mul_2}x2e")
        
        # Input irreps to first layer: just scalars (0e) from embedding
        self.irreps_in = o3.Irreps(f"{mul_0}x0e")
        
        # 2. Backbone Interaction Blocks
        self.layers = nn.ModuleList()
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        
        # We need to bridge from embedding (scalars) to hidden irreps in the first layer
        # InteractionBlock usually takes 'irreps_in' and produces 'irreps_out'.
        # However, it expects 'irreps_in' to match the previous output.
        # We might need an initial Linear or TensorProduct to populate higher ls?
        # Standard e3nn pattern: Start with scalars, interaction mixes in Y(r) to create l>0.
        
        # Layer 0: scalars -> hidden
        self.layers.append(
            InteractionBlock(
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_hidden,
                number_of_radial_basis_functions=10,
                steps=torch.linspace(0.0, r_max, 10),
            )
        )
        
        # Subsequent layers: hidden -> hidden
        for _ in range(num_layers - 1):
             self.layers.append(
                InteractionBlock(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
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
        # Extract features from Absorber State + Context
        # s_a (scalars), c (context), norms(l=1), norms(l=2)
        
        # Calculate input dim for final MLP
        # s_a: mul_0
        # c: mul_0
        # |v_a|^2: mul_1 (one norm per channel if we treat them as independent vectors? Or sum? 
        # User said "The invariant norms of the l=1 features". Usually means norm per multiplicity channel.
        # So mul_1 scalars.
        # |t_a|^2: mul_2 scalars.
        
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
            data: PyG Data object with:
                  - x: (N, 1) or z: (N,) atomic numbers
                  - pos: (N, 3)
                  - edge_index: (2, E)
                  - batch: (N,)
                  - absorber_mask: (N,) boolean
        """
        z = data.z if hasattr(data, 'z') else data.x.squeeze().long()
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
            h = layer(h, edge_vec=vec, edge_attr=edge_sh, edge_length=edge_len, edge_src=edge_src, edge_dst=edge_dst)
            
        # Extract Absorber State for Readout
        # h matches self.irreps_hidden
        # 64x0e + 32x1o + 16x2e
        
        # We need to decompose h back into parts
        # e3nn doesn't always make this trivial if we just have a tensor.
        # But we know the slices because fixed irreps.
        
        idx_0 = 0
        len_0 = 64 # mul_0
        idx_1 = idx_0 + len_0
        len_1 = 32 * 3 # mul_1 * 3
        idx_2 = idx_1 + len_1
        len_2 = 16 * 5 # mul_2 * 5
        
        scalars_all = h[:, idx_0:idx_0+len_0]
        l1_all = h[:, idx_1:idx_1+len_1].reshape(-1, 32, 3) # [N, 32, 3]
        l2_all = h[:, idx_2:idx_2+len_2].reshape(-1, 16, 5) # [N, 16, 5]
        
        # Absorber specific features
        mask = data.absorber_mask
        s_a = scalars_all[mask] # [N_graphs, 64]
        v_a = l1_all[mask]      # [N_graphs, 32, 3]
        t_a = l2_all[mask]      # [N_graphs, 16, 5]
        
        # Norms
        # sq norm = sum squares over component dim
        norm_v = torch.sum(v_a**2, dim=-1) # [N_graphs, 32]
        norm_t = torch.sum(t_a**2, dim=-1) # [N_graphs, 16]
        
        # Context Pooling
        # The pooling layer expects the full node features 'h' and returns aggregated context
        # But our pooling layer as implemented expects 'x' and parses scalars internally.
        # Let's pass 'scalars_all' to it to be safe/simple, or pass 'h' if it handles slicing.
        # My implementation of AbsorberQueryAttention takes 'x' and slices self.n_scalars.
        # If I pass 'h', it will take the first 64 channels, which IS the scalars.
        c = self.pooling(h, mask, batch) # [N_graphs, 64]
        
        # Concatenate
        z_readout = torch.cat([s_a, c, norm_v, norm_t], dim=1) # [64+64+32+16]
        
        # Predict Coefficients
        coeffs = self.readout_mlp(z_readout) # [N_graphs, num_basis]
        
        return coeffs
    
    def predict_spectra(self, data, energy_grid):
        coeffs = self.forward(data) # [B, M]
        basis_matrix = self.basis(energy_grid) # [N_E, M]
        # spectra = basis @ coeffs.T -> [N_E, B] -> transpose -> [B, N_E]
        spectra = torch.matmul(basis_matrix, coeffs.T).T
        return spectra

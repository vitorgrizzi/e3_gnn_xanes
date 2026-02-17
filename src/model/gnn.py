import torch
import torch.nn as nn
from e3nn import o3

from src.model import MultiScaleGaussianBasis, AtomicEmbedding, CustomInteractionBlock, AbsorberQueryAttention


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
                 num_radial=10,
                 basis_scales=[0.1, 0.5, 1.0],
                 emin=-10, emax=50):
        super().__init__()
        
        # Store multiplicities for use in forward()
        self.mul_0 = mul_0
        self.mul_1 = mul_1
        self.mul_2 = mul_2
        
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
                number_of_radial_basis_functions=num_radial,
                steps=torch.linspace(0.0, r_max, num_radial),
            )
        )
        
        # Subsequent layers: hidden -> hidden
        for _ in range(num_layers - 1):
             self.layers.append(
                CustomInteractionBlock(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    number_of_radial_basis_functions=num_radial,
                    steps=torch.linspace(0.0, r_max, num_radial),
                )
             )
             
        # 3. Pooling
        self.pooling = AbsorberQueryAttention(
            irep_in=self.irreps_hidden,
            hidden_dim=mul_0
        )
        
        # 4. Readout Head
        # s_a (mul_0) + context (mul_0) + norm_v (mul_1) + norm_t (mul_2)
        readout_dim = mul_0 + mul_0 + mul_1 + mul_2
        
        self.readout_mlp = nn.Sequential(
            nn.Linear(readout_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_basis)
        )
        
        # Basis
        self.basis = MultiScaleGaussianBasis(
            n_basis=num_basis, emin=emin, emax=emax, scales_ratios=basis_scales
        )
        
    def forward(self, data):
        """
        Args:
            data: PyG Data object with pos, z, edge_index, edge_shift,
                  batch, absorber_mask.  ``edge_shift`` carries the
                  Cartesian PBC displacement so that the true edge
                  vector is ``pos[dst] - pos[src] + edge_shift``.
        """
        # Fallback for data.z vs data.x
        if hasattr(data, 'z') and data.z is not None:
             z = data.z
        else:
             raise ValueError("data.z is None")
             
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        
        # Embed
        h = self.embedding(z)
        
        # Edge attributes  (PBC-aware)
        edge_src, edge_dst = edge_index
        edge_shift = getattr(data, 'edge_shift', torch.zeros_like(pos[edge_src]))
        vec = pos[edge_dst] - pos[edge_src] + edge_shift
        edge_len = vec.norm(dim=1)
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, vec, normalize=True, normalization='component'
        )
        
        # Interaction Blocks
        for layer in self.layers:
            h = layer(
                x=h, edge_attr=edge_sh, edge_length=edge_len,
                edge_src=edge_src, edge_dst=edge_dst,
            )
            
        # Slice features by irrep type using stored multiplicities
        current_idx = 0
        slices = []
        for mul, ir in self.irreps_hidden:
            length = mul * ir.dim
            slices.append((current_idx, current_idx + length))
            current_idx += length
            
        s_range, v_range, t_range = slices[0], slices[1], slices[2]
        
        scalars_all = h[:, s_range[0]:s_range[1]]
        l1_all = h[:, v_range[0]:v_range[1]].reshape(-1, self.mul_1, 3)
        l2_all = h[:, t_range[0]:t_range[1]].reshape(-1, self.mul_2, 5)
        
        # Absorber-specific features
        mask = data.absorber_mask
        s_a = scalars_all[mask] 
        v_a = l1_all[mask]      
        t_a = l2_all[mask]      
        
        # Invariant norms of higher-order features
        norm_v = torch.sum(v_a ** 2, dim=-1)   # [N_graph, mul_1]
        norm_t = torch.sum(t_a ** 2, dim=-1)   # [N_graph, mul_2]
        
        # Context Pooling
        c = self.pooling(h, mask, batch)         # [N_graph, mul_0]
        
        # Concatenate & predict coefficients
        z_readout = torch.cat([s_a, c, norm_v, norm_t], dim=1) 
        coeffs = self.readout_mlp(z_readout)
        
        return coeffs
    
    def predict_spectra(self, data, energy_grid):
        coeffs = self.forward(data)                        # [B, M]
        energy_grid = energy_grid.to(coeffs.device)        # device guard
        basis_matrix = self.basis(energy_grid)              # [N_E, M]
        spectra = torch.matmul(basis_matrix, coeffs.T).T
        return spectra

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate

# Module-level torch_scatter import
try:
    from torch_scatter import scatter
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


class AtomicEmbedding(nn.Module):
    """
    Learnable embedding for atomic numbers.
    Maps integer Z to a vector of size hidden_dim.
    """
    def __init__(self, max_z=100, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(max_z + 1, embedding_dim)
        
    def forward(self, z):
        return self.embedding(z)


class CustomInteractionBlock(nn.Module):
    """
    E(3)-equivariant interaction block with gated non-linearity.

    Performs message passing: m_ij = TP(h_j, Y_ij) * MLP(d_ij),
    aggregates messages, applies residual connection, and gates
    the output (scalar activation on l=0, gated activation on l>0).
    """
    def __init__(self,
                 irreps_in,
                 irreps_out,
                 irreps_sh,
                 number_of_radial_basis_functions,
                 steps=None):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        
        # --- Build Gate irreps ---
        # Gate needs: scalar activations for l=0, and one gate scalar per
        # non-scalar channel (l>0) to modulate them.
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        )
        irreps_nonscalars = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]
        )
        # One gate scalar (0e) per non-scalar multiplicity
        irreps_gates = o3.Irreps(
            [(mul, "0e") for mul, ir in self.irreps_out if ir.l > 0]
        )

        # The TP must produce: scalars + gate_scalars + non-scalars
        self.irreps_tp_out = irreps_scalars + irreps_gates + irreps_nonscalars

        # Gate module
        act_scalars = [torch.nn.functional.silu] * len(irreps_scalars)
        act_gates = [torch.sigmoid] * len(irreps_gates)

        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=act_scalars,
            irreps_gates=irreps_gates,
            act_gates=act_gates,
            irreps_gated=irreps_nonscalars,
        )
        
        # --- Tensor Product ---
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_tp_out,
            shared_weights=False,
            internal_weights=False,
        )
        
        # --- Radial basis + MLP for TP weights ---
        self.num_radial = number_of_radial_basis_functions
        if steps is not None:
            self.register_buffer('radial_centers', steps)
            if len(steps) > 1:
                self.radial_sigma = (steps[-1] - steps[0]) / (len(steps) - 1)
            else:
                self.radial_sigma = 1.0
        else:
            self.radial_sigma = 1.0
            self.register_buffer(
                'radial_centers',
                torch.linspace(0, 5.0, self.num_radial),
            )

        self.fc = FullyConnectedNet(
            [self.num_radial, 64, self.tp.weight_numel],
            torch.nn.functional.silu,
        )
        
        # --- Residual connection ---
        self.has_residual = (self.irreps_in == self.irreps_out)
        self.sc = None
        if self.has_residual:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)
            
        # --- Normalization for stability ---
        self.norm = o3.Norm(self.irreps_out)

    def radial_basis(self, length):
        """Gaussian radial basis: exp(-(d - mu)^2 / 2sigma^2)."""
        d = length.unsqueeze(-1) - self.radial_centers.unsqueeze(0)
        return torch.exp(-(d ** 2) / (2 * self.radial_sigma ** 2))

    def forward(self, x, edge_vec=None, edge_attr=None, edge_length=None,
                edge_src=None, edge_dst=None):
        # 1. Radial embedding → TP weights
        radial = self.radial_basis(edge_length)
        weights = self.fc(radial)
        
        # 2. Message computation via tensor product
        x_j = x[edge_src]
        m_ij = self.tp(x_j, edge_attr, weights)
        
        # 3. Aggregate messages at destination nodes (Mean is more stable than Add)
        if HAS_TORCH_SCATTER:
            m_i = scatter(m_ij, edge_dst, dim=0, dim_size=x.shape[0], reduce='mean')
        else:
            m_i = torch.zeros(x.shape[0], m_ij.shape[1], device=x.device, dtype=x.dtype)
            count = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            m_i.index_add_(0, edge_dst, m_ij)
            count.index_add_(0, edge_dst, torch.ones_like(m_ij[:,:1]))
            m_i = m_i / count.clamp(min=1)

        # 4. Gated non-linearity
        m_i = self.gate(m_i)

        # 5. Residual connection
        if self.sc is not None:
            m_i = m_i + self.sc(x)
            
        # 6. Final normalization per layer
        return self.norm(m_i)

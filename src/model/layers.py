import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate

# torch_scatter is used for highly optimized message 
# aggregation (sum, mean, max, etc.) in message passing
try:
    from torch_scatter import scatter
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


class AtomicEmbedding(nn.Module):
    """
    Learnable embedding for atomic numbers.
    Maps integer Z to a vector of size `embedding_dim`. 
    Returns the initial node embedding for each atom type. 
    """
    def __init__(self, max_z=100, embedding_dim=64):
        super().__init__()
        # Unused entries are never updated and don't hurt learning
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
                 r_max=5.0,
                 dropout=0.1):
        super().__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.dropout = nn.Dropout(dropout)
        
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
        self.r_max = r_max
        
        # Frequencies for spherical bessel functions: n*pi / r_max
        # where n = 1, 2, ..., num_radial
        n_pi = torch.arange(1, self.num_radial + 1, dtype=torch.float32) * torch.pi
        self.register_buffer('bessel_freqs', n_pi / self.r_max)

        self.fc = FullyConnectedNet(
            [self.num_radial, 64, self.tp.weight_numel],
            torch.nn.functional.silu,
        )
        
        # --- Residual connection ---
        self.has_residual = (self.irreps_in == self.irreps_out)
        self.sc = None
        if self.has_residual:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)
            
            

    def radial_basis(self, length):
        """
        Spherical Bessel radial basis with cosine smooth cutoff.
        """
        # Add small epsilon to avoid division by zero at origin
        d = length.unsqueeze(-1) + 1e-8
        
        # 1. Spherical Bessel j_0(k*d) = sin(k*d) / (k*d)
        # We can use torch.sinc(x) = sin(pi*x)/(pi*x), so we pass (k*d)/pi
        k_d = d * self.bessel_freqs.unsqueeze(0)

        # Since torch.sinc expects x to produce sin(pi*x)/(pi*x), we pass x = k_d / pi
        bessel = torch.sinc(k_d / torch.pi)
        
        # 2. Cosine cutoff polynomial: 0.5 * (cos(pi * d / r_max) + 1.0)
        # Evaluates to 1.0 at d=0 and smoothly goes to 0.0 at d=r_max
        cutoff = 0.5 * (torch.cos(torch.pi * d / self.r_max) + 1.0)
        
        # Enforce exact zero outside r_max just in case
        cutoff = cutoff * (d < self.r_max).float()
        
        return bessel*cutoff

    def forward(self, x, edge_attr=None, edge_length=None, edge_src=None, edge_dst=None):
        # 1. Radial embedding -> TP weights
        radial = self.radial_basis(edge_length)
        weights = self.fc(radial)
        
        # 2. Message computation via tensor product
        x_j = x[edge_src]
        m_ij = self.tp(x_j, edge_attr, weights)
        
        # Apply dropout to messages before aggregation
        m_ij = self.dropout(m_ij)
        
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
            
        return m_i

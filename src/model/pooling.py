import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter_softmax, scatter_add


class AbsorberQueryAttention(nn.Module):
    """
    Absorber-Query Attention Pooling.
    
    Uses the absorber's state as a Query to attend to neighbor atoms (Keys/Values).
    Aggregates neighbor features into a single context vector.
    """
    def __init__(self, irep_in, hidden_dim=64):
        super().__init__()
        self.irep_in = o3.Irreps(irep_in)
        
        # Count scalar (0e) channels
        self.n_scalars = 0
        for mul, ir in self.irep_in:
            if ir.l == 0 and ir.p == 1:
                self.n_scalars += mul
        
        if self.n_scalars == 0:
            raise ValueError(
                "Input irreps must contain scalar (0e) features for attention pooling."
            )

        # Attention MLP: (s_absorber, s_neighbor) → score
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * self.n_scalars, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x, absorber_mask, batch):
        """
        Args:
            x: Feature tensor [N_nodes, feature_dim] matching irep_in.
            absorber_mask: Boolean mask [N_nodes], True for absorber atoms.
            batch: Batch index [N_nodes].
            
        Returns:
            context: Pooled context vectors [N_graphs, n_scalars].
        """
        # Extract scalar features (assumed to be first n_scalars channels)
        scalars = x[:, :self.n_scalars]
        
        # Absorber query per graph → expand to all nodes via batch vector
        q_global = scalars[absorber_mask]          # [N_graphs, n_scalars]
        q_expanded = q_global[batch]               # [N_nodes, n_scalars]
        
        # Attention scores
        cat_features = torch.cat([q_expanded, scalars], dim=1)
        e = self.attn_mlp(cat_features)            # [N_nodes, 1]
        
        # Mask out the absorber itself (score = -inf)
        e = e.masked_fill(absorber_mask.unsqueeze(1), -1e9)
        
        # Softmax per graph + weighted sum
        alpha = scatter_softmax(e, batch, dim=0)
        weighted = alpha * scalars
        context = scatter_add(weighted, batch, dim=0)
            
        return context

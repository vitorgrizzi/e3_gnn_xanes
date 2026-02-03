import torch
import torch.nn as nn
from e3nn import o3

class AbsorberQueryAttention(nn.Module):
    """
    Absorber-Query Attention Pooling.
    
    Uses the absorber's state as a Query to attend to neighbor atoms (Keys/Values).
    Aggregates neighbor features into a single context vector.
    """
    def __init__(self, irep_in, hidden_dim=64):
        super().__init__()
        self.irep_in = o3.Irreps(irep_in)
        
        # We assume the scalar part (0e) is the first part of the irreps or extractable.
        # For simplicity, we'll perform attention on the scalar channels found in the input.
        # If the input is mixed irreps, we need to extract scalars.
        
        # Count scalar channels
        self.n_scalars = 0
        for mul, ir in self.irep_in:
            if ir.l == 0 and ir.p == 1: # 0e
                self.n_scalars += mul
        
        if self.n_scalars == 0:
            raise ValueError("Input irreps must contain scalar (0e) features for attention pooling mechanism.")

        # MLP for attention score
        # Inputs: [s_absorber, s_neighbor, distance_embedding, ...]
        # For this implementation, we'll assume the input to compute 'e' 
        # is just the concatenation of (s_absorber, s_neighbor). 
        # Distance info is usually encoded in the edge features previous to this or can be added.
        # To keep it simple and generic: Attention(s_a, s_j)
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * self.n_scalars, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, absorber_mask, batch):
        """
        Args:
            x (torch.Tensor): Feature tensor [N_nodes, feature_dim].
                              Should match irep_in.
            absorber_mask (torch.Tensor): Boolean mask [N_nodes] where True indicates absorber.
            batch (torch.Tensor): Batch index [N_nodes].
            
        Returns:
            context (torch.Tensor): Pooled context vectors [N_graphs, n_scalars].
        """
        # Extract scalar features
        # Assuming typical e3nn layout where scalars are often first or we should slice.
        # For robustness, we should probably rely on the user to ensure scalars are at the start
        # OR use e3nn narrowing. Let's assume scalars are the first n_scalars channels.
        scalars = x[:, :self.n_scalars]
        
        # Separate into absorber and others
        # This is tricky because "others" are variable number. 
        # We need to broadcast absorber 'q' to all 'k' in the same graph.
        
        # Get absorber features for each graph
        # absorber_mask has exactly one True per graph usually.
        # We can use index_select or similar.
        
        # shape: [N_graphs, n_scalars]
        q_global = scalars[absorber_mask]
        
        # Expand q back to node dimension to pair with every node
        # We can align using batch vector
        q_expanded = q_global[batch]
        
        # Compute attention scores e_j
        # cat [q, k] -> [N_nodes, 2*n_scalars]
        cat_features = torch.cat([q_expanded, scalars], dim=1)
        e = self.attn_mlp(cat_features) # [N_nodes, 1]
        
        # Mask out the absorber itself from attending (score = -inf)
        # We want sum_{j != a} alpha_j s_j
        e = e.masked_fill(absorber_mask.unsqueeze(1), -1e9)
        
        # Softmax over the graph
        # We need global softmax per graph.
        # scatter_softmax from torch_scatter is standard for this in PyG.
        # If not available, we can implement logsumexp logic, but let's assume torch_scatter 
        # or implement a simple version if we can't assume dependencies.
        # Given "e3nn" context, usually torch_scatter is present.
        # I'll try to import it, if not, I'll write a manual stable softmax using batch indices.
        
        try:
            from torch_scatter import scatter_softmax, scatter_add
            alpha = scatter_softmax(e, batch, dim=0)
            
            # Weighted sum
            # alpha: [N_nodes, 1]
            # scalars: [N_nodes, n_scalars]
            weighted = alpha * scalars
            
            # Sum over graph
            context = scatter_add(weighted, batch, dim=0)
            
        except ImportError:
            # Fallback for no torch_scatter
            # Note: This is less efficient and stable but works for standard setups
            # Not implementing full fallback to keep it concise, assuming PyG env
            # But let's write a "safe" version just in case for the agent context
            raise ImportError("torch_scatter is required for this pooling layer.")
            
        return context

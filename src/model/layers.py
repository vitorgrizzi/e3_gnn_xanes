import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet

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

# Wrapper for e3nn interaction block logic if needed
# For now we might mainly rely on direct e3nn calls in the main GNN, 
# but it's good to have a helper if we want to enforce specific logic.
# The user specified: TP(h_j, Y) . MLP(d) -> Gate -> ...
# This is physically close to e3nn.nn.models.gate_points_2102.InteractionBlock

def get_mlp_activation(activation_str="silu"):
    if activation_str == "silu":
        return torch.nn.SiLU()
    elif activation_str == "relu":
        return torch.nn.ReLU()
    else:
        return torch.nn.SiLU()

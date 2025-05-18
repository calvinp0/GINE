"""
DimeNet++ Encoder for molecular graphs.
Wraps the PyTorch Geometric DimeNetPlusPlus model to produce graph-level embeddings.
"""
import torch
from torch import nn
from torch_geometric.nn import DimeNetPlusPlus
import numpy as np
import math
# Monkey-patch numpy to include math module for PyG's sph_harm implementation
np.math = math

class DimeNetPPEncoder(nn.Module):
    """
    DimeNet++ encoder that outputs fixed-size graph embeddings.

    Args:
        hidden_channels (int): Number of hidden channels in interaction blocks.
        out_channels (int): Dimension of the output embedding (out_dim).
        num_blocks (int): Number of message-passing blocks.
        num_spherical (int): Number of spherical basis functions.
        num_radial (int): Number of radial basis functions.
        cutoff (float): Distance cutoff for interactions.
        envelope_exponent (int): Exponent of the envelope function.
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels:   int = 128,
        num_blocks:     int = 4,
        num_spherical:  int = 7,
        num_radial:     int = 6,
        cutoff:         float = 5.0,
        envelope_exponent: int = 5,
        dropout:        float = 0.1,
    ):
        super().__init__()
        # Core DimeNet++ model: note signature requires int_emb_size, basis_emb_size, out_emb_channels
        self.model = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=hidden_channels,
            basis_emb_size=hidden_channels,
            out_emb_channels=out_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )
        # Expose output dimension for compatibility
        self.out_dim = out_channels
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        """
        Compute graph embeddings.

        Args:
            data (torch_geometric.data.Data): Batch of graph data with attributes
                - x: [N, F] node features
                - edge_index: [2, E] bond indices
                - edge_attr: [E, D] bond attributes
                - batch: [N] batch assignment

        Returns:
            torch.Tensor: [batch_size, out_channels] graph embeddings
        """
        # Use atomic numbers and positions
        z = data.z
        pos = data.pos
        batch = data.batch
        # DimeNetPlusPlus expects (z, pos, batch)
        embed = self.model(data.z, data.pos, data.batch)
        return self.dropout(embed)
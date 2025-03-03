import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGNN(nn.Module):
    """
    Simplified Graph Neural Network for user similarity matching
    Uses Graph Convolutional Networks (GCN) which are easier to work with locally
    """
    
    def __init__(self, input_dim, hidden_dim=32, embedding_dim=64, num_layers=2):
        super(SimpleGNN, self).__init__()
        self.embedding_dim = embedding_dim
        
        # First GCN layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Middle GCN layers (if needed)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Final GCN layer
        self.conv_final = GCNConv(hidden_dim, embedding_dim)
        
        # Projection layer for similarity calculation
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through the GNN"""
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        
        x = self.conv_final(x, edge_index, edge_weight)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
        
        return x
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Get user embeddings from the GNN"""
        with torch.no_grad():
            return self.forward(x, edge_index, edge_weight)

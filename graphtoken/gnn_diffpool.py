import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, GCNConv, GATConv, TransformerConv, SAGEConv, GINConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.dense import dense_diff_pool
import math

def build_conv(conv_type: str):
    """Return the specific GNN convolution layer as specified by `conv_type`."""
    if conv_type == "GCN":
        return GCNConv
    elif conv_type == "GIN":
        return lambda i, h: GINConv(nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))
    elif conv_type == "GAT":
        return GATConv
    elif conv_type == "TransformerConv":
        return TransformerConv
    elif conv_type == "SAGE":
        return SAGEConv
    # Dense convolutions for use after the first pooling layer
    elif conv_type == "DenseSAGE":
        return DenseSAGEConv
    elif conv_type == "DenseGCN":
        return DenseGCNConv
    else:
        raise KeyError("GNN_TYPE is not a valid convolution type.")

class GNNEncoder(nn.Module):
    """
    A Graph Neural Network Encoder for **SPARSE** graphs, operating on `(x, edge_index)`.
    This is used for the very first layer of the DiffPool model before the graph becomes dense.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, gnn_type="SAGE", dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.act = nn.LeakyReLU()
        
        self.conv_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Build GNN layers
        if n_layers == 1:
            self.conv_layers.append(build_conv(gnn_type)(input_dim, output_dim))
        else:
            self.conv_layers.append(build_conv(gnn_type)(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(n_layers - 2):
                self.conv_layers.append(build_conv(gnn_type)(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.conv_layers.append(build_conv(gnn_type)(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        node_emb = self.conv_layers[-1](x, edge_index)
        return node_emb

class DenseGNNEncoder(nn.Module):
    """
    A Graph Neural Network Encoder for **DENSE** graphs, operating on `(x, adj, mask)`.
    This is used for all hierarchical layers after the initial pooling.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, gnn_type="DenseSAGE", dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.act = nn.LeakyReLU()
        
        self.conv_layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        if n_layers == 1:
            self.conv_layers.append(build_conv(gnn_type)(input_dim, output_dim))
        else:
            self.conv_layers.append(build_conv(gnn_type)(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(n_layers - 2):
                self.conv_layers.append(build_conv(gnn_type)(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.conv_layers.append(build_conv(gnn_type)(hidden_dim, output_dim))

    def forward(self, x, adj, mask=None):
        x_in = x
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, adj, mask)
            # Batch norm needs reshaping for dense format (Batch, Nodes, Features)
            x = self.bns[i](x.transpose(1, 2)).transpose(1, 2)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        node_emb = self.conv_layers[-1](x, adj, mask)
        return node_emb


class HierarchicalDiffPool(nn.Module):
    """
    A hierarchical Differentiable Pooling model with a configurable number of pooling layers.
    The number of clusters at each layer is determined by a ratio of the previous layer's nodes.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, pool_ratio=0.8, gnn_type="SAGE", n_layers=2, max_nodes_per_graph=23, dropout=0.0):
        super().__init__()
        
        self.gnn_embed_layers = nn.ModuleList()
        self.gnn_pool_layers = nn.ModuleList()

        # Calculate the number of clusters for each pooling layer
        num_clusters = [math.ceil(pool_ratio * max_nodes_per_graph)]
        for _ in range(1, n_layers):
            num_clusters.append(math.ceil(pool_ratio * num_clusters[-1]))

        # --- Layer 0: Operates on sparse graph ---
        # GNN to compute node embeddings for the first pooling
        self.gnn_embed_layers.append(GNNEncoder(input_dim, hidden_dim, hidden_dim, 1, gnn_type, dropout))
        # GNN to compute cluster assignments for the first pooling
        self.gnn_pool_layers.append(GNNEncoder(input_dim, hidden_dim, num_clusters[0], 1, gnn_type, dropout))
        
        # --- Subsequent Layers: Operate on dense, pooled graphs ---
        dense_gnn_type = 'Dense' + gnn_type
        current_dim = hidden_dim
        for i in range(1, n_layers):
            self.gnn_embed_layers.append(DenseGNNEncoder(current_dim, hidden_dim, hidden_dim, 1, dense_gnn_type, dropout))
            self.gnn_pool_layers.append(DenseGNNEncoder(current_dim, hidden_dim, num_clusters[i], 1, dense_gnn_type, dropout))
            current_dim = hidden_dim
            
        # Final linear layer to get the desired output dimension
        self.final_linear = nn.Linear(current_dim, output_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
            
        total_link_loss = 0
        total_entropy_loss = 0

        # --- Initial Sparse to Dense Pooling ---
        gnn_embed = self.gnn_embed_layers[0]
        gnn_pool = self.gnn_pool_layers[0]
        
        x_embed = gnn_embed(x, edge_index)
        s = gnn_pool(x, edge_index)
        s = F.softmax(s, dim=-1)

        # Convert sparse representations to dense for the diff_pool operation
        x_embed_dense, mask = to_dense_batch(x_embed, batch)
        s_dense, _ = to_dense_batch(s, batch)
        adj_dense = to_dense_adj(edge_index, batch)
        
        # The first diff_pool call
        x, adj, link_loss, entropy_loss = dense_diff_pool(x_embed_dense, adj_dense, s_dense, mask=mask)
        total_link_loss += link_loss
        total_entropy_loss += entropy_loss

        # --- Subsequent Dense to Dense Pooling Layers ---
        for i in range(1, len(self.gnn_pool_layers)):
            gnn_embed = self.gnn_embed_layers[i]
            gnn_pool = self.gnn_pool_layers[i]
            
            # The mask is no longer needed as the graph is now dense and unpadded
            x_embed = gnn_embed(x, adj)
            s = gnn_pool(x, adj)
            s = F.softmax(s, dim=-1)
            
            x, adj, link_loss, entropy_loss = dense_diff_pool(x_embed, adj, s)
            total_link_loss += link_loss
            total_entropy_loss += entropy_loss

        # --- Final Readout ---
        # Take the sum of node features in the final coarsened graph
        # graph_embedding = torch.sum(x, dim=1)
        # graph_embedding = self.act(graph_embedding)
        output = self.final_linear(x)
        
        return output.squeeze(0)


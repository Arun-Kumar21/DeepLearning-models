import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import to_dense_batch

class ConvModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x, batch):
        # x: (total_nodes, hidden_dim)
        # batch: (total_nodes,) — which graph each node belongs to
        x_dense, mask = to_dense_batch(x, batch)  # (B, max_nodes, hidden_dim)
        x_dense = x_dense.permute(0, 2, 1)        # (B, hidden_dim, max_nodes)
        x_conv  = self.relu(self.conv(x_dense))   # (B, hidden_dim, max_nodes)
        x_conv  = x_conv.permute(0, 2, 1)         # (B, max_nodes, hidden_dim)

        # mask out padding, then mean pool
        x_conv  = x_conv * mask.unsqueeze(-1).float()
        graph_vec = x_conv.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        return self.relu(self.linear(graph_vec))   # (B, hidden_dim)
    
class DevignModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=200, num_steps=6):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ggnn       = GatedGraphConv(hidden_dim, num_layers=num_steps)
        self.conv       = ConvModule(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x          = self.input_proj(data.x)          # (N, hidden_dim)
        x          = self.ggnn(x, data.edge_index)    # (N, hidden_dim) — message passing
        graph_vec  = self.conv(x, data.batch)          # (B, hidden_dim)
        return self.classifier(graph_vec).squeeze(-1)  # (B,)  raw logits


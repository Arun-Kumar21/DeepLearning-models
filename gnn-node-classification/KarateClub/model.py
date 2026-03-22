from torch_geometric import nn
import torch


class Model(torch.nn.Module):
    def __init__ (self, num_features, num_classes):
        super().__init__()
        self.backbone = nn.Sequential('x, edge_index', [
            nn.GCNConv(num_features, 3),
            torch.nn.Linear(3, num_classes)
        ])

    def forward(self, x):
        return self.backbone(x)

        
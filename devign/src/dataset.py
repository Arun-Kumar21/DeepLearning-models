import os, torch
from torch.utils.data import Dataset

class DevignDataset(Dataset):
    def __init__(self, graph_dir, ids):
        self.graph_dir = graph_dir
        self.ids = ids  

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        gid = self.ids[idx]
        return torch.load(os.path.join(self.graph_dir, f"graph_{gid}.pt"))
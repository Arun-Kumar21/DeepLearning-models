from torch_geometric.loader import DataLoader

from dataset import DevignDataset

train_ids = list(range(0, 20000))
val_ids = list(range(20000, 23000))
test_ids = list(range(23000, 27318))

train_set = DevignDataset("data/graphs", train_ids)
val_set = DevignDataset("data/graphs", val_ids)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

import torch
from torch_geometric.loader import DataLoader
from model import DevignModel

# load just 8 graphs
from dataset import DevignDataset 
tiny = DevignDataset("data/graphs", ids=list(range(8)))
loader = DataLoader(tiny, batch_size=8)
batch  = next(iter(loader))

model     = DevignModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

for step in range(200):
    model.train()
    optimizer.zero_grad()
    logits = model(batch)
    loss   = criterion(logits, batch.y.float())
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")

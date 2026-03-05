from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj 
from torch_geometric.utils import to_networkx 
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

dataset = KarateClub()

print(dataset)

print('Number of graphs:', len(dataset))
print('Number of features:', dataset.num_features)
print('Number of classes:', dataset.num_classes)


print(f'Graph: {dataset[0]}')


data =  dataset[0]
print(f'Shape of data point: {data.x.shape}')
print (data.x)


print("Adjancey Matrix:")

adj = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print (f'Adjancey shape: {adj.shape}')
print (adj)

print ("Target Groups:")
y = data.y
print (y.shape)
print (y)


print(f'train_mask = {data.train_mask.shape}')
print (data.train_mask)


print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')


G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12, 12))
plt.axis('off')


nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=42),
                 node_size=800,
                 node_color=data.y,
                 cmap="hsv",
                 vmin=-2,
                 vmax=3,
                 width=0.8,
                 edge_color='grey',
                 font_size=14
                 )

plt.show()

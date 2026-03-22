import torch
from torch_geometric.data import Data 
from gensim.models import Word2Vec
import numpy as np


EDGE_TYPES = {"AST": 0, "CFG": 1, "DDG": 2, "CDG": 3}

def build_graph(cpg_json:dict, label:int, w2v: Word2Vec) -> Data:
    nodes = cpg_json["nodes"]
    edges = cpg_json["edges"]

    node_features = []
    for node in nodes:
        code = node.get("code", "")
        tokens = code.split()
        if tokens:
            vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
            feat = np.mean(vecs, axios=0) if vecs else np.zeros(100)

        else:
            feat = np.zeros(100)

        node_features.append(feat)


    x = torch.tensor(np.array(node_features), dtype=torch.float)

    src_list, dst_list, edge_type = [], [], []
    node_id_map = {n["id"]: i for i, n in enumerate(nodes)}
    
    for edge in edges:
        s = node_id_map.get(edge['src'])
        d = node_id_map.get(edge['dst'])
        t = EDGE_TYPES.get(edge['type'], 0)
        if s is not None and d is not None:
            src_list.append(s)
            dst_list.append(d)
            edge_type.append(t)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_type, dtype=torch.long)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
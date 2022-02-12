# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:23:23 2021

@author: mirza
"""

import torch
from torch_geometric.data import Data
print(1)
edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
 #making nodes
 #Node feature matrix with shape [num_nodes, num_node_features]
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index) 
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
  
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# print (dataset[0])
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# for batch in loader:
#      print(batch)
#      print(batch.num_graphs) 
# data = dataset[0]
# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
#print(f'Number of training nodes: {data.train_mask.sum()}')
#print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected=True)
#helper function, check colab notebook mentioned in endnotes
#visualize(G, color=data.y) 
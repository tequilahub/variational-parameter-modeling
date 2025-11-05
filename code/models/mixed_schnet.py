# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet

class MixedSchNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.emb_dim = hidden_dim

        self.backbone1 = SchNet()

        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, z, batch=None, pos=None, edges=None):
        out = self.process_backbone(self.backbone1, self.mlp1, z, self.process_input(pos, edges[0]), batch)
        return out

    def process_backbone(self, backbone, mlp, z, pos, batch):
        # multi = factor for each atom/2 how many angles
        h = backbone.embedding(z)
        edge_index, edge_weight = backbone.interaction_graph(pos, batch)
        edge_attr = backbone.distance_expansion(edge_weight)

        for interaction in backbone.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        node_embeddings = backbone.lin1(h)
        atoms, _ = node_embeddings.shape

        pairwise_emb = node_embeddings.reshape(-1, 2 * node_embeddings.size(1))

        angle_preds = mlp(pairwise_emb)
        angle_preds = angle_preds.reshape(-1)
        return angle_preds

    def process_input(self, pos, edges):
        index_order = []
        for pair in edges:
            for idx in pair:
                index_order.append(idx.item())

        seen = set()
        ordered_unique_indices = []
        for idx in index_order:
            if idx not in seen:
                seen.add(idx)
                ordered_unique_indices.append(idx)

        pos_sorted = pos[ordered_unique_indices]
        return pos_sorted
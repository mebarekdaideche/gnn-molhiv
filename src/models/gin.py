# Référence : Xu et al., ICLR 2019 — https://arxiv.org/abs/1810.00826
# GIN est théoriquement aussi expressif que le test de Weisfeiler-Lehman.

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

from src.models.base_gnn import BaseGNN
from src.models.encoders import AtomEncoder


def build_mlp(in_dim, out_dim):
    # MLP à 2 couches requis par GIN pour son expressivité maximale
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class GIN(BaseGNN):

    def __init__(self, num_layers=5, emb_dim=300, num_tasks=1, dropout=0.5):
        super().__init__(num_layers, emb_dim, num_tasks, dropout)

        self.atom_encoder = AtomEncoder(emb_dim)

        self.convs = nn.ModuleList([
            GINConv(nn=build_mlp(emb_dim, emb_dim), train_eps=True)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(emb_dim) for _ in range(num_layers)
        ])

    def forward(self, data):
        h = self.atom_encoder(data.x)

        for i, conv in enumerate(self.convs):
            h = conv(h, data.edge_index)
            h = self.bns[i](h)
            if i < self.num_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)

        h = global_mean_pool(h, data.batch)
        h = self.dropout(h)
        return self.graph_pred_linear(h)

# Référence : Kipf & Welling, ICLR 2017 — https://arxiv.org/abs/1609.02907
# Limitation : GCNConv ignore les features d'arêtes. Voir gine.py pour les inclure.

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from src.models.base_gnn import BaseGNN
from src.models.encoders import AtomEncoder


class GCN(BaseGNN):

    def __init__(self, num_layers=5, emb_dim=300, num_tasks=1, dropout=0.5):
        super().__init__(num_layers, emb_dim, num_tasks, dropout)

        self.atom_encoder = AtomEncoder(emb_dim)

        self.convs = nn.ModuleList([
            GCNConv(emb_dim, emb_dim) for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(emb_dim) for _ in range(num_layers)
        ])

    def forward(self, data):
        h = self.atom_encoder(data.x)

        for i, conv in enumerate(self.convs):
            h = conv(h, data.edge_index)  # GCNConv ignore edge_attr
            h = self.bns[i](h)
            if i < self.num_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)

        h = global_mean_pool(h, data.batch)
        h = self.dropout(h)
        return self.graph_pred_linear(h)

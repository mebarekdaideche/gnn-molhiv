# Référence : Hu et al., ICLR 2020 — https://arxiv.org/abs/1905.12265
# GINE étend GIN en intégrant les features d'arêtes :
#   message(u→v) = ReLU(h_u + BondEncoder(e_{u,v}))

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

from src.models.base_gnn import BaseGNN
from src.models.encoders import AtomEncoder, BondEncoder
from src.models.gin import build_mlp


class GINE(BaseGNN):

    def __init__(self, num_layers=5, emb_dim=300, num_tasks=1, dropout=0.5):
        super().__init__(num_layers, emb_dim, num_tasks, dropout)

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        self.convs = nn.ModuleList([
            GINEConv(nn=build_mlp(emb_dim, emb_dim), train_eps=True, edge_dim=emb_dim)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(emb_dim) for _ in range(num_layers)
        ])

    def forward(self, data):
        h        = self.atom_encoder(data.x)
        edge_emb = self.bond_encoder(data.edge_attr)

        for i, conv in enumerate(self.convs):
            h = conv(h, data.edge_index, edge_emb)
            h = self.bns[i](h)
            if i < self.num_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)

        h = global_mean_pool(h, data.batch)
        h = self.dropout(h)
        return self.graph_pred_linear(h)

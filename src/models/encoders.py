import torch
import torch.nn as nn


class AtomEncoder(nn.Module):
    """
    9 features catégorielles d'atome → vecteur dense de dimension emb_dim.
    Un embedding par feature, somme des 9 embeddings (standard OGB).

    Features OGB : atomic_num, chirality, degree, formal_charge,
                   num_hs, num_radical_e, hybridization, is_aromatic, is_in_ring
    """

    DIMS = [119, 4, 12, 12, 10, 6, 6, 2, 2]

    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(d, emb_dim) for d in self.DIMS
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x):
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x[:, i])
        return out


class BondEncoder(nn.Module):
    """
    3 features catégorielles de liaison → vecteur dense de dimension emb_dim.
    Features OGB : bond_type (4), bond_stereo (6), is_conjugated (2)
    """

    DIMS = [4, 6, 2]

    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(d, emb_dim) for d in self.DIMS
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, edge_attr):
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(edge_attr[:, i])
        return out

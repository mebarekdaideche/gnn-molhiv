import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class BaseGNN(nn.Module):
    """
    Classe parente pour GCN, GIN, GINE.
    Sous-classes doivent définir self.convs, self.bns et forward().
    """

    def __init__(self, num_layers=5, emb_dim=300, num_tasks=1, dropout=0.5):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers doit être >= 2")

        self.num_layers = num_layers
        self.emb_dim    = emb_dim
        self.num_tasks  = num_tasks
        self.dropout    = nn.Dropout(p=dropout)

        # Logits bruts — BCEWithLogitsLoss applique sigmoid en interne
        self.graph_pred_linear = nn.Linear(emb_dim, num_tasks)

        self.convs = None
        self.bns   = None

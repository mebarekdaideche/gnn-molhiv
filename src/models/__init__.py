from src.models.gcn  import GCN
from src.models.gin  import GIN
from src.models.gine import GINE

MODEL_REGISTRY = {
    "gcn":  GCN,
    "gin":  GIN,
    "gine": GINE,
}

__all__ = ["GCN", "GIN", "GINE", "MODEL_REGISTRY"]

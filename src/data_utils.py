# src/data_utils.py
"""
Chargement et préparation du dataset OGB-MolHIV.

Ce module gère :
- Le téléchargement automatique du dataset via OGB
- La création des DataLoaders train/val/test avec le split officiel
- L'affichage des statistiques du dataset

Note sur le split :
    OGB-MolHIV utilise un scaffold split (division par squelette moléculaire),
    plus difficile et plus réaliste qu'un split aléatoire.
    On utilise TOUJOURS le split officiel pour que nos scores soient
    comparables au leaderboard OGB.
"""

import torch
from torch_geometric.loader import DataLoader

# ── Compatibilité PyTorch >= 2.6 ─────────────────────────────────────────────
# Depuis PyTorch 2.6, torch.load() a weights_only=True par défaut.
# OGB appelle torch.load() en interne sans ce paramètre, ce qui provoque :
#   _pickle.UnpicklingError: Unsupported global: DataEdgeAttr
# Ce patch force weights_only=False uniquement pour les appels sans argument explicite.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
# ─────────────────────────────────────────────────────────────────────────────

from ogb.graphproppred import PygGraphPropPredDataset


def load_dataset(root: str = "dataset") -> PygGraphPropPredDataset:
    """
    Télécharge (si nécessaire) et charge le dataset OGB-MolHIV.

    Le dataset est mis en cache dans le dossier `root/` après le premier
    téléchargement (~3 Mo). Les appels suivants sont instantanés.

    Args:
        root : dossier de stockage local du dataset

    Returns:
        Le dataset PyG complet (41 127 graphes moléculaires)
    """
    print(f"[Data] Chargement du dataset ogbg-molhiv depuis '{root}'...")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    print(f"[Data] Dataset chargé : {len(dataset)} molécules")
    return dataset


def get_dataloaders(
    dataset: PygGraphPropPredDataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> tuple:
    """
    Crée les DataLoaders train/val/test à partir du split officiel OGB.

    Le scaffold split est récupéré directement depuis OGB — on ne le
    recalcule jamais soi-même pour garantir la cohérence avec le leaderboard.

    Args:
        dataset     : le dataset OGB-MolHIV chargé
        batch_size  : nombre de graphes par batch (32 est un bon défaut)
        num_workers : workers pour le chargement parallèle
                      (0 = main process, plus simple sur Windows/Mac)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx["train"]]
    val_dataset   = dataset[split_idx["valid"]]
    test_dataset  = dataset[split_idx["test"]]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def print_dataset_stats(dataset: PygGraphPropPredDataset) -> None:
    """
    Affiche les statistiques essentielles du dataset.
    Utile pour comprendre les données avant de construire le modèle.
    """
    split_idx = dataset.get_idx_split()

    print("\n" + "=" * 55)
    print("  STATISTIQUES — OGB-MolHIV")
    print("=" * 55)
    print(f"  Nombre total de molécules : {len(dataset):>10,}")
    print(f"  Train                     : {len(split_idx['train']):>10,}")
    print(f"  Validation                : {len(split_idx['valid']):>10,}")
    print(f"  Test                      : {len(split_idx['test']):>10,}")
    print("-" * 55)

    sample = dataset[0]
    print(f"  Features par atome (nœud)   : {sample.x.shape[1]:>8}")
    print(f"  Features par liaison (arête): {sample.edge_attr.shape[1]:>8}")
    print(f"  Nombre de tâches (labels)   : {dataset.num_tasks:>8}")
    print("-" * 55)

    train_labels = dataset[split_idx["train"]].y
    n_pos        = (train_labels == 1).sum().item()
    n_neg        = (train_labels == 0).sum().item()
    n_nan        = (train_labels != train_labels).sum().item()
    total_valid  = n_pos + n_neg

    print(f"  [Train] Actifs  (y=1) : {n_pos:>8,}  ({100 * n_pos / total_valid:.1f}%)")
    print(f"  [Train] Inactifs(y=0) : {n_neg:>8,}  ({100 * n_neg / total_valid:.1f}%)")
    print(f"  [Train] Labels NaN    : {n_nan:>8,}  (filtrés à l'entraînement)")
    print("=" * 55 + "\n")

    print("  Exemple d'un graphe moléculaire :")
    print(f"  {sample}")
    print(f"  → {sample.num_nodes} atomes, "
          f"{sample.num_edges // 2} liaisons (× 2 bidirectionnel)\n")

# src/utils.py
"""
Utilitaires généraux du projet.
Contient : gestion des seeds, du device, des logs, des chemins.
"""

import os
import random
import logging
import numpy as np
import torch


# ─────────────────────────────────────────────
#  REPRODUCTIBILITÉ
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fixe toutes les sources d'aléatoire pour garantir la reproductibilité.
    À appeler au tout début de chaque script d'entraînement.

    PyTorch, NumPy et Python ont chacun leur propre générateur aléatoire —
    il faut tous les fixer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Retourne automatiquement le meilleur device disponible.
    GPU NVIDIA > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU détecté : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Aucun GPU détecté, entraînement sur CPU.")
    return device


# ─────────────────────────────────────────────
#  GESTION DES CHEMINS
# ─────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """
    Crée le dossier (et ses parents) s'il n'existe pas encore.
    Retourne le chemin pour pouvoir l'utiliser en one-liner.

    Exemple :
        log_path = ensure_dir("results/gcn/") + "metrics.csv"
    """
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Configure un logger qui écrit à la fois dans le terminal et dans un fichier.

    Args:
        name     : nom du logger (en général __name__ du module appelant)
        log_file : chemin vers le fichier de log (optionnel)

    Returns:
        Un logger Python standard configuré.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            ensure_dir(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────
#  COMPTAGE DES PARAMÈTRES
# ─────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """
    Compte le nombre de paramètres entraînables d'un modèle.
    Utile pour comparer la complexité des architectures.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

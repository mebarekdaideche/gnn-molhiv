# src/run_experiments.py
"""
Script d'expérimentation automatisée.

Lance une grille d'expériences (modèles × seeds) et compile
les résultats dans results/results_summary.csv.

Usage :
    # Mode rapide : 3 modèles × 1 seed × 30 epochs (~1h CPU)
    python -m src.run_experiments --mode quick

    # Mode complet : 3 modèles × 3 seeds × 100 epochs (~6h CPU / ~45min GPU)
    python -m src.run_experiments --mode full

    # Un seul modèle avec plusieurs seeds
    python -m src.run_experiments --models gine --seeds 42 0 1

    # Surcharger le nombre d'epochs
    python -m src.run_experiments --mode full --epochs 50
"""

import argparse
import subprocess
import sys
from itertools import product

from src.utils import ensure_dir


# ─────────────────────────────────────────────
#  GRILLES D'EXPÉRIENCES
# ─────────────────────────────────────────────

# Mode "quick" : vérifie que tout marche et donne une tendance
QUICK_CONFIG = {
    "models":     ["gcn", "gin", "gine"],
    "seeds":      [42],
    "epochs":     30,
    "lr":         0.001,
    "num_layers": 5,
    "emb_dim":    300,
    "dropout":    0.5,
    "batch_size": 32,
}

# Mode "full" : pour les résultats publiables (mean ± std sur 3 seeds)
FULL_CONFIG = {
    "models":     ["gcn", "gin", "gine"],
    "seeds":      [42, 0, 1],
    "epochs":     100,
    "lr":         0.001,
    "num_layers": 5,
    "emb_dim":    300,
    "dropout":    0.5,
    "batch_size": 32,
}


def run_single_experiment(model: str, seed: int, config: dict) -> bool:
    """
    Lance une expérience via subprocess.
    Retourne True si terminée sans erreur.
    """
    cmd = [
        sys.executable, "-m", "src.train",
        "--model",      model,
        "--seed",       str(seed),
        "--epochs",     str(config["epochs"]),
        "--lr",         str(config["lr"]),
        "--num_layers", str(config["num_layers"]),
        "--emb_dim",    str(config["emb_dim"]),
        "--dropout",    str(config["dropout"]),
        "--batch_size", str(config["batch_size"]),
    ]

    print(f"\n{'=' * 60}")
    print(f"  {model.upper()} | seed={seed} | epochs={config['epochs']}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_all_experiments(config: dict) -> None:
    """Lance toutes les combinaisons modèle × seed."""
    ensure_dir("results/")

    experiments = list(product(config["models"], config["seeds"]))
    total       = len(experiments)
    failed      = []

    print(f"\nTotal : {total} expérience(s) à lancer")
    print(f"Modèles : {config['models']}")
    print(f"Seeds   : {config['seeds']}")
    print(f"Epochs  : {config['epochs']}\n")

    for i, (model, seed) in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] {model.upper()}, seed={seed}")
        success = run_single_experiment(model, seed, config)
        if not success:
            failed.append((model, seed))
            print(f"  ⚠️  Échec : {model}, seed={seed}")

    print(f"\n{'=' * 60}")
    print(f"  TERMINÉ : {total - len(failed)}/{total} réussies")
    if failed:
        print(f"  Échecs  : {failed}")
    print(f"  Résultats → results/results_summary.csv")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grille d'expériences GNN sur OGB-MolHIV"
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Sous-ensemble de modèles (ex: --models gcn gine)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds spécifiques (ex: --seeds 42 0 1)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Surcharger le nombre d'epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = QUICK_CONFIG.copy() if args.mode == "quick" else FULL_CONFIG.copy()

    if args.models:
        config["models"] = args.models
    if args.seeds:
        config["seeds"] = args.seeds
    if args.epochs:
        config["epochs"] = args.epochs

    run_all_experiments(config)

# src/evaluate.py
"""
Évaluation d'un ou plusieurs modèles sauvegardés sur OGB-MolHIV.

Usage :
    # Évaluer un checkpoint spécifique
    python -m src.evaluate --checkpoint saved_models/best_gine_lr0.001_layers5_dim300_seed42.pt

    # Évaluer tous les checkpoints dans saved_models/
    python -m src.evaluate --all

Ce script est utile pour :
- Vérifier les performances sans réentraîner
- Comparer plusieurs checkpoints de façon systématique
- Détecter l'overfitting (gap train AUC vs test AUC)
"""

import os
import glob
import argparse
import torch
from ogb.graphproppred import Evaluator

from src.data_utils import load_dataset, get_dataloaders
from src.models import MODEL_REGISTRY
from src.utils import set_seed, get_device, get_logger


# ─────────────────────────────────────────────
#  ÉVALUATION D'UN CHECKPOINT
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str, device: torch.device, logger) -> dict:
    """
    Charge un checkpoint et évalue le modèle sur train/val/test.

    Args:
        checkpoint_path : chemin vers le fichier .pt
        device          : cpu ou cuda
        logger          : logger Python

    Returns:
        Dictionnaire des scores {model, seed, epoch, train_auc, val_auc,
                                  test_auc, overfit_gap}
    """
    logger.info(f"Chargement : {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args       = checkpoint["args"]

    logger.info(f"  Modèle   : {args['model'].upper()}")
    logger.info(f"  Layers   : {args['num_layers']} | emb_dim : {args['emb_dim']}")
    logger.info(f"  Seed     : {args['seed']}")
    logger.info(f"  Epoch    : {checkpoint['epoch']}")

    set_seed(args["seed"])

    dataset = load_dataset(root=args.get("data_root", "dataset"))
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=args.get("batch_size", 32)
    )

    ModelClass = MODEL_REGISTRY[args["model"]]
    model = ModelClass(
        num_layers=args["num_layers"],
        emb_dim=args["emb_dim"],
        num_tasks=dataset.num_tasks,
        dropout=args["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    evaluator = Evaluator(name="ogbg-molhiv")

    def _eval(loader):
        y_true_list, y_pred_list = [], []
        for batch in loader:
            batch = batch.to(device)
            pred  = model(batch)
            y_true_list.append(batch.y.detach().cpu())
            y_pred_list.append(pred.detach().cpu())
        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        return evaluator.eval({
            "y_true": y_true.numpy(),
            "y_pred": y_pred.numpy()
        })["rocauc"]

    logger.info("  Évaluation en cours...")
    train_auc = _eval(train_loader)
    val_auc   = _eval(val_loader)
    test_auc  = _eval(test_loader)
    gap       = round(train_auc - test_auc, 4)

    logger.info(f"  Train AUC   : {train_auc:.4f}")
    logger.info(f"  Val   AUC   : {val_auc:.4f}")
    logger.info(f"  Test  AUC   : {test_auc:.4f}")
    logger.info(
        f"  Gap train-test : {gap:.4f} "
        f"{'⚠️  overfitting possible' if gap > 0.05 else '✅ OK'}"
    )

    return {
        "checkpoint":  os.path.basename(checkpoint_path),
        "model":       args["model"],
        "seed":        args["seed"],
        "epoch":       checkpoint["epoch"],
        "train_auc":   round(train_auc, 4),
        "val_auc":     round(val_auc, 4),
        "test_auc":    round(test_auc, 4),
        "overfit_gap": gap,
    }


# ─────────────────────────────────────────────
#  ÉVALUATION DE TOUS LES CHECKPOINTS
# ─────────────────────────────────────────────

def evaluate_all_checkpoints(device, logger) -> None:
    """Évalue tous les .pt dans saved_models/ et affiche un bilan."""
    checkpoints = glob.glob("saved_models/best_*.pt")
    if not checkpoints:
        logger.warning("Aucun checkpoint trouvé dans saved_models/")
        return

    logger.info(f"Trouvé {len(checkpoints)} checkpoint(s)\n")
    all_results = []

    for ckpt_path in sorted(checkpoints):
        try:
            results = evaluate_checkpoint(ckpt_path, device, logger)
            all_results.append(results)
            print()
        except Exception as e:
            logger.error(f"Erreur sur {ckpt_path} : {e}")

    if not all_results:
        return

    # Bilan final
    print("=" * 75)
    print("  BILAN — Tous les checkpoints")
    print("=" * 75)
    print(f"  {'Checkpoint':<48} {'Val':>7} {'Test':>7} {'Gap':>7}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: -x["test_auc"]):
        flag = " ⚠️" if r["overfit_gap"] > 0.05 else "  ✅"
        print(
            f"  {r['checkpoint']:<48} "
            f"{r['val_auc']:>7.4f} "
            f"{r['test_auc']:>7.4f} "
            f"{r['overfit_gap']:>7.4f}{flag}"
        )
    print("=" * 75)

    best = max(all_results, key=lambda x: x["test_auc"])
    print(f"\n  🏆 Meilleur modèle : {best['checkpoint']}")
    print(f"     Test AUC = {best['test_auc']:.4f}\n")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation de checkpoints GNN")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Chemin vers un .pt spécifique")
    group.add_argument("--all", action="store_true", help="Évaluer tous les checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger(__name__)
    device = get_device()

    if args.all:
        evaluate_all_checkpoints(device, logger)
    else:
        evaluate_checkpoint(args.checkpoint, device, logger)

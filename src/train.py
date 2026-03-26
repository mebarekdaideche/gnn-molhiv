import os
import argparse
import time
import csv
from datetime import datetime

import torch
import torch.nn as nn
from ogb.graphproppred import Evaluator

from src.data_utils import load_dataset, get_dataloaders, print_dataset_stats
from src.models import MODEL_REGISTRY
from src.utils import set_seed, get_device, ensure_dir, get_logger, count_parameters


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn    = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred   = model(batch)
        labels = batch.y.float()

        # OGB-MolHIV contient quelques labels manquants
        is_labeled = ~torch.isnan(labels)
        if is_labeled.sum() == 0:
            continue

        loss = loss_fn(pred[is_labeled], labels[is_labeled])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        y_true.append(batch.y.detach().cpu())
        y_pred.append(model(batch).detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    return evaluator.eval({
        "y_true": y_true.numpy(),
        "y_pred": y_pred.numpy(),
    })


def write_csv(rows, filepath):
    """Écrit une liste de dicts dans un CSV (crée le fichier si absent)."""
    if not rows:
        return
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_training(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = (
        f"{args.model}_lr{args.lr}_layers{args.num_layers}"
        f"_dim{args.emb_dim}_seed{args.seed}"
    )
    log_dir = ensure_dir(f"results/runs/{run_name}_{timestamp}/")
    logger  = get_logger(__name__, log_file=os.path.join(log_dir, "training.log"))

    logger.info(f"Run : {run_name}")
    logger.info(f"Args : {vars(args)}")

    set_seed(args.seed)
    device = get_device()

    dataset = load_dataset(root=args.data_root)
    print_dataset_stats(dataset)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=args.batch_size
    )

    if args.model not in MODEL_REGISTRY:
        raise ValueError(
            f"Modèle '{args.model}' inconnu. Disponibles : {list(MODEL_REGISTRY)}"
        )

    model = MODEL_REGISTRY[args.model](
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        num_tasks=dataset.num_tasks,
        dropout=args.dropout,
    ).to(device)

    logger.info(f"Modèle : {args.model.upper()} — {count_parameters(model):,} paramètres")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )
    evaluator = Evaluator(name="ogbg-molhiv")

    best_val_auc = 0.0
    final_test_auc = 0.0  # test AUC au moment du meilleur val
    best_epoch   = 0
    history      = []

    logger.info(f"Entraînement : {args.epochs} époques")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_auc    = evaluate(model, val_loader,  evaluator, device)["rocauc"]
        test_auc   = evaluate(model, test_loader, evaluator, device)["rocauc"]

        # Train AUC coûteux : on l'évalue toutes les 10 époques seulement
        if epoch % 10 == 0 or epoch == 1:
            train_auc = evaluate(model, train_loader, evaluator, device)["rocauc"]
        else:
            train_auc = float("nan")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            logger.info(f"  LR : {prev_lr:.6f} → {new_lr:.6f}")

        logger.info(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train: {train_auc:.4f} | "
            f"Val: {val_auc:.4f} | "
            f"Test: {test_auc:.4f}"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_auc": train_auc, "val_auc": val_auc, "test_auc": test_auc,
        })

        # Sélection sur val uniquement — jamais test
        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            final_test_auc = test_auc
            best_epoch     = epoch

            model_path = os.path.join(ensure_dir("saved_models/"), f"best_{run_name}.pt")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc":              val_auc,
                "test_auc":             test_auc,
                "args":                 vars(args),
            }, model_path)

    duration = time.time() - t0

    logger.info("=" * 55)
    logger.info(f"  Meilleure Val AUC : {best_val_auc:.4f}  (epoch {best_epoch})")
    logger.info(f"  Test AUC associé  : {final_test_auc:.4f}")
    logger.info(f"  Durée             : {duration / 60:.1f} min")
    logger.info("=" * 55)

    ensure_dir("results/")
    write_csv([{
        "timestamp":     timestamp,
        "model":         args.model,
        "num_layers":    args.num_layers,
        "emb_dim":       args.emb_dim,
        "lr":            args.lr,
        "dropout":       args.dropout,
        "batch_size":    args.batch_size,
        "seed":          args.seed,
        "best_epoch":    best_epoch,
        "best_val_auc":  round(best_val_auc, 4),
        "best_test_auc": round(final_test_auc, 4),
        "n_params":      count_parameters(model),
        "duration_min":  round(duration / 60, 1),
    }], "results/results_summary.csv")

    # Historique : une seule ouverture de fichier
    write_csv(history, os.path.join(log_dir, "history.csv"))

    logger.info("Résultats -> results/results_summary.csv")
    return best_val_auc, final_test_auc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraînement GNN sur OGB-MolHIV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",        type=str,   default="gcn", choices=list(MODEL_REGISTRY))
    parser.add_argument("--num_layers",   type=int,   default=5)
    parser.add_argument("--emb_dim",      type=int,   default=300)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--data_root",    type=str,   default="dataset")
    parser.add_argument("--config",       type=str,   default=None)

    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        parser.set_defaults(**config)
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    run_training(parse_args())

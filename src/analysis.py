"""
Analyse qualitative d'un modele sauvegarde.

Produit :
1. Courbes de convergence (train/val AUC par epoque)
2. Courbe ROC + distribution des scores de confiance
3. Matrice de confusion + analyse des faux negatifs
4. Visualisation RDKit des molecules (si disponible)

Usage :
    python -m src.analysis --checkpoint saved_models/best_gine_lr0.001_layers5_dim300_seed42.pt
    python -m src.analysis --checkpoint saved_models/best_gine_...pt --no_molecules
"""

import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.data_utils import load_dataset, get_dataloaders
from src.models import MODEL_REGISTRY
from src.utils import get_device, set_seed, ensure_dir, get_logger


@torch.no_grad()
def get_predictions(checkpoint_path, device):
    """Charge un checkpoint et retourne les predictions sur val + test."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args       = checkpoint["args"]
    set_seed(args["seed"])

    dataset = load_dataset(root=args.get("data_root", "dataset"))
    _, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=args.get("batch_size", 32)
    )

    model = MODEL_REGISTRY[args["model"]](
        num_layers=args["num_layers"],
        emb_dim=args["emb_dim"],
        num_tasks=dataset.num_tasks,
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def collect(loader):
        y_true_list, y_logit_list = [], []
        for batch in loader:
            batch = batch.to(device)
            y_true_list.append(batch.y.detach().cpu())
            y_logit_list.append(model(batch).detach().cpu())
        y_true   = torch.cat(y_true_list,  dim=0).numpy().flatten()
        y_logits = torch.cat(y_logit_list, dim=0).numpy().flatten()
        y_probs  = torch.sigmoid(torch.tensor(y_logits)).numpy()
        return y_true, y_probs

    val_true,  val_probs  = collect(val_loader)
    test_true, test_probs = collect(test_loader)

    return {
        "model_name": args["model"].upper(),
        "seed":       args["seed"],
        "val_true":   val_true,
        "val_probs":  val_probs,
        "test_true":  test_true,
        "test_probs": test_probs,
        "args":       args,
        "dataset":    dataset,
    }


def plot_learning_curves(history_dir="results/runs/", output_path="results/learning_curves.png"):
    history_files = glob.glob(f"{history_dir}/**/history.csv", recursive=True)

    if not history_files:
        print(f"[Warning] Aucun history.csv trouve dans {history_dir}")
        return

    n     = len(history_files)
    width = min(6 * n, 30)  # cap a 30 pouces
    fig, axes = plt.subplots(1, n, figsize=(width, 4), squeeze=False)

    for idx, hist_file in enumerate(sorted(history_files)):
        ax    = axes[0][idx]
        df    = pd.read_csv(hist_file)
        label = os.path.basename(os.path.dirname(hist_file))[:40]

        ax.plot(df["epoch"], df["train_auc"], label="Train AUC", color="#4C9BE8", linewidth=1.5)
        ax.plot(df["epoch"], df["val_auc"],   label="Val AUC",   color="#F5A623", linewidth=1.5)

        best = df.loc[df["val_auc"].idxmax()]
        ax.axvline(best["epoch"], color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.annotate(
            f"Best\n{best['val_auc']:.4f}",
            xy=(best["epoch"], best["val_auc"]),
            xytext=(8, -20), textcoords="offset points",
            fontsize=8, color="gray",
        )

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC-AUC")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Courbes de convergence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Courbes de convergence -> {output_path}")
    plt.show()


def plot_roc_and_distribution(preds):
    ensure_dir("results/")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1  = axes[0]
    mask = ~np.isnan(preds["test_true"])
    fpr, tpr, _ = roc_curve(preds["test_true"][mask], preds["test_probs"][mask])
    roc_val     = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color="#7ED321", lw=2, label=f"ROC AUC = {roc_val:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatoire (0.5)")
    ax1.fill_between(fpr, tpr, alpha=0.08, color="#7ED321")
    ax1.set_title(f"Courbe ROC — {preds['model_name']} (Test Set)", fontsize=12)
    ax1.set_xlabel("Taux de Faux Positifs (FPR)")
    ax1.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2   = axes[1]
    probs = preds["test_probs"][mask]
    true  = preds["test_true"][mask]

    ax2.hist(probs[true == 0], bins=40, density=True, alpha=0.6,
             color="#4C9BE8", label=f"Inactives (n={int((true==0).sum()):,})",
             edgecolor="white", linewidth=0.3)
    ax2.hist(probs[true == 1], bins=40, density=True, alpha=0.7,
             color="#E84C4C", label=f"Actives   (n={int((true==1).sum()):,})",
             edgecolor="white", linewidth=0.3)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Seuil = 0.5")

    ax2.set_title("Distribution des scores de confiance", fontsize=12)
    ax2.set_xlabel("Score predit (apres sigmoid)")
    ax2.set_ylabel("Densite")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"{preds['model_name']} | Seed {preds['seed']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"results/roc_distribution_{preds['model_name'].lower()}_seed{preds['seed']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Viz] ROC + distribution -> {out}")
    plt.show()


def analyze_errors(preds, threshold=0.5):
    mask  = ~np.isnan(preds["test_true"])
    probs = preds["test_probs"][mask]
    true  = preds["test_true"][mask].astype(int)
    pred  = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    print("\n" + "=" * 58)
    print(f"  ANALYSE DES ERREURS — {preds['model_name']} (seuil={threshold})")
    print("=" * 58)
    print(f"  Vrais Positifs  (TP) : {tp:>7,}")
    print(f"  Vrais Negatifs  (TN) : {tn:>7,}")
    print(f"  Faux Positifs   (FP) : {fp:>7,}")
    print(f"  Faux Negatifs   (FN) : {fn:>7,}")
    print("-" * 58)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"  Precision : {prec:.4f}")
    print(f"  Rappel    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print("=" * 58)
    print(f"\n  Le modele rate {fn} molecules actives sur {tp + fn}.")
    print(f"  En drug discovery, les FN sont les erreurs les plus couteuses.\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1     = axes[0]
    cm_disp = np.array([[tn, fp], [fn, tp]])
    im      = ax1.imshow(cm_disp, cmap="Blues")

    labels_grid = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_disp[i, j] > cm_disp.max() / 2 else "black"
            ax1.text(j, i, f"{labels_grid[i][j]}\n{cm_disp[i,j]:,}",
                     ha="center", va="center", fontsize=13, fontweight="bold", color=color)

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Predit Inactif", "Predit Actif"])
    ax1.set_yticklabels(["Reel Inactif", "Reel Actif"])
    ax1.set_title(f"Matrice de confusion (seuil={threshold})", fontsize=12)
    plt.colorbar(im, ax=ax1)

    ax2     = axes[1]
    fn_mask = (true == 1) & (pred == 0)
    tp_mask = (true == 1) & (pred == 1)

    if fn_mask.sum() > 0:
        ax2.hist(probs[fn_mask], bins=20, alpha=0.7, color="#E84C4C",
                 label=f"Faux Negatifs (n={fn_mask.sum()})", edgecolor="white")
    if tp_mask.sum() > 0:
        ax2.hist(probs[tp_mask], bins=20, alpha=0.7, color="#7ED321",
                 label=f"Vrais Positifs (n={tp_mask.sum()})", edgecolor="white")

    ax2.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
                label=f"Seuil = {threshold}")
    ax2.set_title("Scores des molecules actives (test set)", fontsize=12)
    ax2.set_xlabel("Score de confiance predit")
    ax2.set_ylabel("Nombre de molecules")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Analyse des erreurs — {preds['model_name']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"results/error_analysis_{preds['model_name'].lower()}_seed{preds['seed']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[Viz] Analyse des erreurs -> {out}")
    plt.show()


def visualize_molecules(preds, n_show=4, output_path="results/molecule_examples.png"):
    """Visualise des exemples de molécules bien et mal classifiées (nécessite RDKit)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        print("[Warning] RDKit non disponible — conda install -c conda-forge rdkit")
        return

    try:
        smiles_df = pd.read_csv("dataset/ogbg_molhiv/mapping/mol.csv.gz")
    except FileNotFoundError:
        print("[Warning] Fichier SMILES non trouve — visualisation ignoree.")
        return

    dataset      = preds["dataset"]
    test_indices = dataset.get_idx_split()["test"]

    mask  = ~np.isnan(preds["test_true"])
    probs = preds["test_probs"][mask]
    true  = preds["test_true"][mask].astype(int)
    pred  = (probs >= 0.5).astype(int)

    tp_idx    = np.where((true == 1) & (pred == 1))[0]
    fn_idx    = np.where((true == 1) & (pred == 0))[0]
    tp_sorted = tp_idx[np.argsort(-probs[tp_idx])][:n_show]
    fn_sorted = fn_idx[np.argsort( probs[fn_idx])][:n_show]

    def get_mols(indices, category):
        mols, labels = [], []
        for idx in indices:
            global_idx = test_indices[idx].item()
            # smiles_df a un RangeIndex 0..N-1 aligné sur les mol_id OGB
            smiles = smiles_df["smiles"].iloc[global_idx]
            mol    = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
                labels.append(f"{category}\nscore={probs[idx]:.3f}")
        return mols, labels

    all_mols, all_labels = [], []
    for mols, labels in [get_mols(tp_sorted, "TP"), get_mols(fn_sorted, "FN")]:
        all_mols   += mols
        all_labels += labels

    if not all_mols:
        print("[Warning] Aucune molecule a afficher.")
        return

    ensure_dir(os.path.dirname(output_path))
    img = Draw.MolsToGridImage(
        all_mols, molsPerRow=n_show, subImgSize=(300, 250), legends=all_labels
    )
    img.save(output_path)
    print(f"[Viz] Molecules -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse qualitative")
    parser.add_argument("--checkpoint",   type=str,   required=True)
    parser.add_argument("--history_dir",  type=str,   default="results/runs/")
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--no_molecules", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    logger = get_logger(__name__)
    device = get_device()

    logger.info("=== Analyse qualitative ===")

    preds = get_predictions(args.checkpoint, device)

    plot_learning_curves(args.history_dir)
    plot_roc_and_distribution(preds)
    analyze_errors(preds, threshold=args.threshold)

    if not args.no_molecules:
        visualize_molecules(preds)

    logger.info("=== Analyse terminee — resultats dans results/ ===")

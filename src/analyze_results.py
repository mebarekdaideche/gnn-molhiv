"""
Synthèse des résultats depuis results/results_summary.csv.
Produit un tableau mean ± std par modèle et un graphique de comparaison.

Usage :
    python -m src.analyze_results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.utils import ensure_dir


def load_results(filepath="results/results_summary.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            "Lance d'abord : python -m src.run_experiments"
        )
    return pd.read_csv(filepath)


def compute_summary(df):
    summary = (
        df.groupby("model")
        .agg(
            n_runs       = ("seed",         "count"),
            val_mean     = ("best_val_auc",  "mean"),
            val_std      = ("best_val_auc",  "std"),
            test_mean    = ("best_test_auc", "mean"),
            test_std     = ("best_test_auc", "std"),
            avg_params   = ("n_params",      "mean"),
            avg_time_min = ("duration_min",  "mean"),
        )
        .round(4)
        .reset_index()
    )

    def fmt(mean, std):
        return f"{mean:.4f}" if pd.isna(std) else f"{mean:.4f} ± {std:.4f}"

    summary["val_display"]  = summary.apply(lambda r: fmt(r.val_mean,  r.val_std),  axis=1)
    summary["test_display"] = summary.apply(lambda r: fmt(r.test_mean, r.test_std), axis=1)
    return summary


def print_summary_table(summary):
    print("\n" + "=" * 72)
    print("  RÉSULTATS — OGB-MolHIV (Scaffold Split)")
    print("=" * 72)
    print(f"  {'Modèle':<8} {'Val AUC':>22} {'Test AUC':>22} {'Runs':>5}")
    print("-" * 72)

    for _, r in summary.sort_values("test_mean", ascending=False).iterrows():
        print(
            f"  {r['model'].upper():<8} "
            f"{r['val_display']:>22} "
            f"{r['test_display']:>22} "
            f"{int(r['n_runs']):>5}"
        )

    print("-" * 72)
    print("  Références leaderboard OGB officiel :")
    print(f"  {'GCN':<8} {'—':>22} {'~0.7606':>22}")
    print(f"  {'GIN':<8} {'—':>22} {'~0.7558':>22}")
    print(f"  {'GINE':<8} {'—':>22} {'~0.7740':>22}")
    print("=" * 72 + "\n")


def plot_comparison(df, summary, output_path="results/model_comparison.png"):
    ensure_dir(os.path.dirname(output_path))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison GNN — OGB-MolHIV (Scaffold Split)", fontsize=14, fontweight="bold")

    COLORS    = {"GCN": "#4C9BE8", "GIN": "#F5A623", "GINE": "#7ED321"}
    ORDER     = ["GCN", "GIN", "GINE"]
    sum_s     = summary.copy()
    sum_s["model_up"] = sum_s["model"].str.upper()
    sum_s     = sum_s.set_index("model_up").reindex(
        [m for m in ORDER if m in sum_s["model_up"].values]
    ).reset_index()

    ax1        = axes[0]
    models     = sum_s["model_up"]
    test_means = sum_s["test_mean"]
    test_stds  = sum_s["test_std"].fillna(0)

    bars = ax1.bar(
        models, test_means,
        yerr=test_stds,
        color=[COLORS.get(m, "#AAAAAA") for m in models],
        edgecolor="black", linewidth=0.8,
        capsize=6, width=0.5,
        error_kw={"elinewidth": 1.5},
    )

    ax1.axhline(0.7606, color="#4C9BE8", linestyle="--", linewidth=1, alpha=0.6, label="GCN OGB ref.")
    ax1.axhline(0.7740, color="#7ED321", linestyle="--", linewidth=1, alpha=0.6, label="GINE OGB ref.")

    for bar, mean, std in zip(bars, test_means, test_stds):
        label = f"{mean:.4f}" if std == 0 else f"{mean:.4f}\n±{std:.4f}"
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax1.set_ylim(max(0.0, test_means.min() - 0.05), min(1.0, test_means.max() + 0.07))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax1.set_title("Test ROC-AUC par architecture", fontsize=12)
    ax1.set_ylabel("ROC-AUC")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    for name in df["model"].unique():
        sub   = df[df["model"] == name]
        color = COLORS.get(name.upper(), "#AAAAAA")
        ax2.scatter(
            sub["duration_min"], sub["best_test_auc"],
            label=name.upper(), color=color, s=120, edgecolors="black", linewidths=0.8, zorder=3,
        )
        for _, row in sub.iterrows():
            ax2.annotate(
                f"s={int(row['seed'])}",
                (row["duration_min"], row["best_test_auc"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7, color=color,
            )

    ax2.set_title("Performance vs Temps d'entraînement", fontsize=12)
    ax2.set_xlabel("Durée (minutes)")
    ax2.set_ylabel("Test ROC-AUC")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Graphique -> {output_path}")
    plt.show()


if __name__ == "__main__":
    df      = load_results()
    summary = compute_summary(df)

    print_summary_table(summary)
    plot_comparison(df, summary)

    out = "results/final_summary.csv"
    summary[["model", "n_runs", "val_display", "test_display",
             "avg_params", "avg_time_min"]].to_csv(out, index=False)
    print(f"[Results] Tableau final -> {out}")

# gnn-molhiv

Comparaison de GCN, GIN et GINE sur OGB-MolHIV — prédiction d'activité anti-VIH
de molécules. Réalisé en L3 dans le cadre d'une candidature M1 IA/Data Science.

## Résultats (3 seeds, scaffold split officiel OGB)

| Modèle | Test ROC-AUC    | Référence OGB |
|--------|-----------------|---------------|
| GINE   | 0.7744 ± 0.0030 | ~0.7740       |
| GIN    | 0.7631 ± 0.0032 | ~0.7558       |
| GCN    | 0.7606 ± 0.0015 | ~0.7606       |

Environnement : CPU, PyTorch 2.1, PyG 2.4, 100 epochs, seeds 42/0/1.
GINE dépasse GIN de +1.1 pt en intégrant les features des liaisons chimiques
dans le message passing — information ignorée par GCN et GIN standard.

![Comparaison des modèles](results/model_comparison.png)

### Détail par seed

| Modèle | Seed 42 | Seed 0 | Seed 1 |
|--------|---------|--------|--------|
| GINE   | 0.7748  | 0.7712 | 0.7771 |
| GIN    | 0.7634  | 0.7598 | 0.7661 |
| GCN    | 0.7589  | 0.7612 | 0.7618 |

## Dataset

[OGB-MolHIV](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) — 41 127 molécules
du programme AIDS Antiviral Screen du NIH. Tâche : classification binaire
(actif/inactif contre le VIH), ~3.5 % de positifs. Métrique : ROC-AUC.

Chaque molécule est un graphe : nœuds = atomes (9 features), arêtes = liaisons
(3 features : type, stéréochimie, conjugaison).

Split : scaffold (les molécules de test ont des squelettes chimiques absents du
train — plus difficile qu'un split aléatoire, résultats comparables au leaderboard).

## Installation

```bash
conda create -n gnn-molhiv python=3.10 -y
conda activate gnn-molhiv

# PyTorch CPU
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
# PyTorch GPU CUDA 11.8 :
# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric
pip install -r requirements.txt
python check_install.py
```

Le dataset (~3 Mo) est téléchargé automatiquement dans `dataset/` au premier
lancement de `train.py`.

**Google Colab :**
```python
!pip install torch_geometric ogb
```

## Utilisation

```bash
# Entraîner un modèle
python -m src.train --model gine --epochs 100 --seed 42

# Via fichier de config YAML
python -m src.train --config configs/gine_config.yaml

# Test rapide (5 epochs)
python -m src.train --model gcn --epochs 5

# Toutes les expériences : 3 modèles × 3 seeds × 100 epochs
python -m src.run_experiments --mode full

# Mode rapide : 3 modèles × 1 seed × 30 epochs
python -m src.run_experiments --mode quick

# Synthèse des résultats + graphiques
python -m src.analyze_results

# Réévaluer un checkpoint
python -m src.evaluate --checkpoint saved_models/best_gine_lr0.001_layers5_dim300_seed42.pt
python -m src.evaluate --all

# Analyse qualitative (ROC, erreurs, molécules)
python -m src.analysis --checkpoint saved_models/best_gine_lr0.001_layers5_dim300_seed42.pt
```

## Architecture

```
src/
  models/
    base_gnn.py    — pooling + tête de classification partagés
    encoders.py    — AtomEncoder et BondEncoder
    gcn.py         — GCNConv
    gin.py         — GINConv + MLP interne
    gine.py        — GINEConv + BondEncoder
  train.py         — boucle d'entraînement
  data_utils.py    — chargement OGB, DataLoaders
  evaluate.py      — réévaluation de checkpoints
  run_experiments.py — grille automatisée
  analyze_results.py — synthèse statistique
  analysis.py      — ROC, erreurs, visualisation moléculaire
configs/           — hyperparamètres YAML par modèle
results/           — graphiques et CSV de résultats
notebooks/         — exploration et analyse interactive
```

## Modèles

**GCN** (Kipf & Welling, ICLR 2017) — agrégation par moyenne pondérée des voisins,
normalisée par les degrés. Ignore les features d'arêtes. Sert de baseline.

**GIN** (Xu et al., ICLR 2019) — agrégation par somme + MLP 2 couches + paramètre ε
appris. Théoriquement aussi expressif que le test de Weisfeiler-Lehman.

**GINE** (Hu et al., ICLR 2020) — étend GIN en intégrant les features de liaison
dans chaque message : `ReLU(h_voisin + BondEncoder(e_{u,v}))`.

Composants partagés : `AtomEncoder` (9 features → emb_dim), global mean pooling,
couche linéaire finale. `BondEncoder` (3 features → emb_dim) uniquement pour GINE.

## Hyperparamètres

| Paramètre          | Valeur                                        |
|--------------------|-----------------------------------------------|
| Couches GNN        | 5                                             |
| emb_dim            | 300                                           |
| Dropout            | 0.5                                           |
| Optimiseur         | Adam, lr=0.001                                |
| LR Scheduler       | ReduceLROnPlateau (patience=10, factor=0.5)   |
| Batch size         | 32                                            |
| Epochs             | 100                                           |
| Loss               | BCEWithLogitsLoss                             |
| Sélection modèle   | Val AUC uniquement (test regardé une seule fois) |

## Analyse qualitative

![ROC et distribution](results/roc_distribution_gine.png)

GINE (seed=42, test set) : TP=85, TN=3677, FP=123, FN=50.
Précision=0.409, Rappel=0.630, F1=0.496.
Les faux négatifs (molécules actives manquées) se concentrent près du seuil 0.5 —
incertitude sur des familles moléculaires absentes du train (scaffold split).

![Analyse des erreurs](results/error_analysis_gine.png)

![Courbes de convergence](results/learning_curves.png)

## Limitations

- 3 seeds seulement — les intervalles de confiance restent larges.
- Pas de tuning d'hyperparamètres : les valeurs utilisées sont celles recommandées
  par le leaderboard OGB, pas optimisées sur ce run.
- Entraînement sur CPU (~11-15 min/run). La reproductibilité bit-à-bit n'est pas
  garantie entre machines différentes.
- La confusion matrix et le F1 utilisent un seuil fixe à 0.5 sur un dataset très
  déséquilibré (3.5 % positifs) — ces métriques sont moins pertinentes que le ROC-AUC.
- `torch.load` est monkey-patché pour corriger une incompatibilité OGB/PyTorch 2.6.
  À retirer quand OGB aura corrigé le bug en amont.

## Références

- Hu et al., *Open Graph Benchmark*, NeurIPS 2020 — [arXiv:2005.00687](https://arxiv.org/abs/2005.00687)
- Kipf & Welling, *Semi-Supervised Classification with GCNs*, ICLR 2017 — [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019 — [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- Hu et al., *Strategies for Pre-training GNNs*, ICLR 2020 — [arXiv:1905.12265](https://arxiv.org/abs/1905.12265)
- Fey & Lenssen, *Fast Graph Representation Learning with PyG*, ICLR-W 2019 — [arXiv:1903.02428](https://arxiv.org/abs/1903.02428)

---

*Auteur : Mebarek — Université Côte d'Azur, L3 Informatique*

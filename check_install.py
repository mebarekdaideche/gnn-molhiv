# check_install.py
# Script de vérification de l'installation
# Exécuter une seule fois après l'installation : python check_install.py

import sys


def check(module_name, import_name=None):
    import_name = import_name or module_name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "version inconnue")
        print(f"  ✅ {module_name:30s} {version}")
    except ImportError as e:
        print(f"  ❌ {module_name:30s} MANQUANT — {e}")


print(f"\nPython : {sys.version}\n")
print("=== Vérification des dépendances ===\n")

check("PyTorch",           "torch")
check("PyTorch Geometric", "torch_geometric")
check("OGB",               "ogb")
check("NumPy",             "numpy")
check("Pandas",            "pandas")
check("Scikit-learn",      "sklearn")
check("Matplotlib",        "matplotlib")
check("Seaborn",           "seaborn")
check("tqdm",              "tqdm")
check("PyYAML",            "yaml")

try:
    check("RDKit", "rdkit")
except Exception:
    print(f"  ⚠️  {'RDKit':30s} non installé (optionnel — visualisation molécules)")

import torch

print(f"\nCUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
else:
    print("(entraînement sur CPU — plus lent mais fonctionnel)")

print("\n=== Test PyTorch Geometric ===")
try:
    from torch_geometric.data import Data

    x          = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    data       = Data(x=x, edge_index=edge_index)
    print(f"  ✅ Graphe test créé : {data}")
except Exception as e:
    print(f"  ❌ Erreur PyG : {e}")

print("\n=== Test OGB ===")
try:
    from ogb.graphproppred import PygGraphPropPredDataset  # noqa: F401
    print("  ✅ Import OGB réussi")
    print("  ℹ️  Le dataset sera téléchargé (~3 Mo) au premier lancement de train.py")
except Exception as e:
    print(f"  ❌ Erreur OGB : {e}")

print("\n=== Fin de la vérification ===\n")

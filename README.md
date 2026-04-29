# 🧪 Pick-a-pKa

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/Chemistry-RDKit-green.svg)](https://www.rdkit.org/)

**Pick-a-pKa** is a unified Python framework for high-accuracy pKa prediction. It provides a single, clean API to access two state-of-the-art Graph Neural Network (GNN) architectures: **MolGpKa** and **pKaLearn**.


---

## 🚀 Key Features

*   **Unified API**: Switch between backends (`molgpka` or `pkalearn`) with a single string argument.
*   **Intelligent Featurization**: Automatic conversion from SMILES/molecules to graph tensors.
*   **Macroscopic Ladders**: (pKaLearn) Iteratively discover sequential pKa values as a molecule deprotonates.
*   **Microstate Distribution**: Calculate the fractional abundance of species at any given pH.
*   **Beautiful Visualization**: Generate publication-quality SVG/PNG plots of microspecies distributions and pKa-annotated molecules.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/pick-a-pka.git
cd pick-a-pka

# Install dependencies
pip install -e .
```

---

## 🛠 Quick Start

```python
from pick_a_pka import PKaPredictor
from rdkit import Chem

# 1. Initialize the predictor (defaults to MolGpKa)
mdl = PKaPredictor(backend='pkalearn', device='cpu')

# 2. Predict pKa values
smiles = "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC"
results = mdl.predict_pka(smiles)

# 3. View the deprotonation ladder (pKaLearn specific)
for site in results:
    print(f"pKa: {site['pka']:.2f} | Site Index: {site['center']}")
```

---

## 🧠 Choosing Your Backend

| Feature | **MolGpKa** | **pKaLearn** |
| :--- | :--- | :--- |
| **Primary Strength** | Fast, robust per-atom scan | Iterative macroscopic ladders |
| **Return Format** | Atom-index dictionary | Sequential list of states |
| **Best For** | High-throughput screening | Complex polyprotic molecules |
| **Iterative?** | No (Single round) | Yes (Recursive) |

---

## 🧪 Advanced Usage

### 1. Sequential Deprotonation (pKaLearn)
The `pkalearn` backend identifies a site, deprotonates it, and re-evaluates the resulting molecule to find the next pKa in the ladder.

```python
mdl = PKaPredictor(backend='pkalearn')
ladder = mdl.predict_pka("OC(=O)c1ccccc1NC(=O)c1ccc(c2ccccc2)cc1")

# Output: 
# [{'smiles': '...', 'pka': 3.12, 'center': 0}, ...]
```

### 2. Microstate Abundance at specific pH
Determine the dominant protonation state of a drug-like molecule at physiological pH (7.4).

```python
mdl = PKaPredictor(backend='resources')
micro = mdl.predict_microstates("CC(=O)O", pH=7.4)

print(f"Dominant pKa: {micro['pka']}")
# Access the RDKit molecule of the major species
major_mol = micro['major_state'] 
```

---

## 📊 Visualization

Pick-a-pka includes a powerful drawing engine to visualize how a molecule’s ionization changes across the pH scale.

### Microspecies Distribution Plot
Generate a plot showing the distribution of all ionization states from pH 0 to 14.

```python
from pick_a_pka.backends.molgpka import plot_microspecies_distribution
from rdkit import Chem

mol = Chem.MolFromSmiles("c1ccc(C(C(=O)O)N)cc1")
# Returns a PIL Image or SVG string
img = plot_microspecies_distribution(mol, vector=False)
img.show()
```

### pKa Annotation
Draw the molecule with predicted pKa values callouts.

```python
from pick_a_pka.backends.molgpka import draw_pka

svg_text = draw_pka(mol, vector=True)
```

---

## 📖 Citation

If you use the pKaLearn backend in your research, please cite:

> **Development of a pKa predictor (pKaLearn) by leveraging teaching experience to improve machine learning**,
> Jérôme Genzling, Ziling Luo, Benjamin Weiser & Nicolas Moitessier,
> *Communications Chemistry*, **2026**,
> DOI: 10.1038/s42004-026-01983-y

If you use the MolGpKa backend, please cite:

> **MolGpka: A Web Server for Small Molecule pKa Prediction Using a Graph-Convolutional Neural Network**,
> Xiaolin Pan, Hao Wang, Cuiyu Li, John Z. H. Zhang, and Changge Ji,
> *Journal of Chemical Information and Modeling*, **2021** *61*(7), 3159-3165,
> DOI: 10.1021/acs.jcim.1c00075


---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

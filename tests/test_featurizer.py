"""Unit tests for the featurizers of both backends.
These test the graph-construction layer in isolation, without running inference.
"""
import pytest
import torch
from rdkit import Chem

from pick_a_pka.backends.molgpka.featurizer import (
    get_atom_features,
    get_bond_pair,
    get_ionization_aid,
    mol_to_graph as molgpka_mol_to_graph,
)
from pick_a_pka.backends.pkalearn.featurizer import (
    mol_to_graph as pkalearn_mol_to_graph,
    from_acid_to_base,
)
from pick_a_pka.backends.pkalearn.model import PkaLearnModel

ACETIC_ACID_MOL = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
NEUTRAL_ACETIC = Chem.MolFromSmiles("CC(=O)O")
CONFIG = PkaLearnModel.DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# MolGpKa featurizer
# ---------------------------------------------------------------------------

class TestMolGpKaFeaturizer:
    def test_get_atom_features_returns_list_per_atom(self):
        features = get_atom_features(ACETIC_ACID_MOL, 0)
        assert len(features) == ACETIC_ACID_MOL.GetNumAtoms()

    def test_atom_features_are_numeric(self):
        features = get_atom_features(ACETIC_ACID_MOL, 0)
        for row in features:
            for val in row:
                assert isinstance(val, (int, float, bool))

    def test_get_bond_pair_symmetric(self):
        src, dst = get_bond_pair(NEUTRAL_ACETIC)
        assert len(src) == len(dst)
        # Each bond appears twice (undirected)
        assert len(src) == NEUTRAL_ACETIC.GetNumBonds() * 2

    def test_mol_to_graph_returns_data_object(self):
        data = molgpka_mol_to_graph(ACETIC_ACID_MOL, 0)
        assert data is not None
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")

    def test_mol_to_graph_tensor_shapes(self):
        data = molgpka_mol_to_graph(ACETIC_ACID_MOL, 0)
        assert data.x.shape[0] == ACETIC_ACID_MOL.GetNumAtoms()
        assert data.edge_index.shape[0] == 2

    def test_get_ionization_aid_finds_carboxyl(self):
        acid_sites, _ = get_ionization_aid(ACETIC_ACID_MOL)
        # Carboxylic OH should be flagged as acidic
        assert len(acid_sites) >= 1

    def test_get_ionization_aid_acid_only(self):
        acid_sites = get_ionization_aid(ACETIC_ACID_MOL, acid_or_base="acid")
        assert isinstance(acid_sites, list)

    def test_get_ionization_aid_base_only(self):
        base_sites = get_ionization_aid(ACETIC_ACID_MOL, acid_or_base="base")
        assert isinstance(base_sites, list)

    def test_get_ionization_aid_invalid_mol_raises(self):
        from pick_a_pka.core.exceptions import InvalidMoleculeError
        with pytest.raises(InvalidMoleculeError):
            get_ionization_aid(None)


# ---------------------------------------------------------------------------
# pKaLearn featurizer
# ---------------------------------------------------------------------------

class TestPkaLearnFeaturizer:
    def test_mol_to_graph_returns_data_or_none(self):
        data = pkalearn_mol_to_graph(NEUTRAL_ACETIC, 0, CONFIG)
        # Returns None if the center cannot be featurized, Data otherwise
        assert data is None or hasattr(data, "x")

    def test_mol_to_graph_tensor_dtype(self):
        # Try each atom as center; at least one should succeed
        for idx in range(NEUTRAL_ACETIC.GetNumAtoms()):
            data = pkalearn_mol_to_graph(NEUTRAL_ACETIC, idx, CONFIG)
            if data is not None:
                assert data.x.dtype == torch.float32
                break

    def test_from_acid_to_base_removes_proton(self):
        """from_acid_to_base on a neutral carboxyl oxygen should give fc=-1."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        # Oxygen index: find the OH oxygen (has 1 implicit H)
        oh_idx = next(
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and a.GetTotalNumHs() == 1
        )
        import copy
        mol_copy = copy.deepcopy(mol)
        found, mol_dep, smi = from_acid_to_base(mol_copy, oh_idx)
        assert found is True
        assert mol_dep.GetAtomWithIdx(oh_idx).GetFormalCharge() == -1

    def test_from_acid_to_base_returns_valid_smiles(self):
        mol = Chem.MolFromSmiles("CC(=O)O")
        oh_idx = next(
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and a.GetTotalNumHs() == 1
        )
        import copy
        found, _, smi = from_acid_to_base(copy.deepcopy(mol), oh_idx)
        assert found
        assert smi != "none"
        assert Chem.MolFromSmiles(smi) is not None

    def test_from_acid_to_base_no_proton_returns_false(self):
        """Carbonyl oxygen has no H; deprotonation should fail."""
        mol = Chem.MolFromSmiles("CC(=O)O")
        import copy
        carbonyl_idx = next(
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and a.GetTotalNumHs() == 0
        )
        found, _, _ = from_acid_to_base(copy.deepcopy(mol), carbonyl_idx)
        assert found is False

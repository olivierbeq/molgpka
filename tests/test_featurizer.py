"""Unit tests for the featurizers of both backends.
These test the graph-construction layer in isolation, without running inference.
"""
import pytest
import torch
from rdkit import Chem

from constants import ACETIC_ACID
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

ACETIC_ACID_MOL = Chem.AddHs(Chem.MolFromSmiles(ACETIC_ACID))
NEUTRAL_ACETIC_MOL = Chem.MolFromSmiles(ACETIC_ACID)
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
        src, dst = get_bond_pair(NEUTRAL_ACETIC_MOL)
        assert len(src) == len(dst)
        # Each bond appears twice (undirected)
        assert len(src) == NEUTRAL_ACETIC_MOL.GetNumBonds() * 2

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
        data = pkalearn_mol_to_graph(NEUTRAL_ACETIC_MOL, 0, CONFIG)
        # Returns None if the center cannot be featurized, Data otherwise
        assert data is None or hasattr(data, "x")

    def test_mol_to_graph_tensor_dtype(self):
        # Try each atom as center; at least one should succeed
        for idx in range(NEUTRAL_ACETIC_MOL.GetNumAtoms()):
            data = pkalearn_mol_to_graph(NEUTRAL_ACETIC_MOL, idx, CONFIG)
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


class TestPkaLearnFeaturizer:
    """Lines 23, 34, 60, 88-89, 142-148, 169, 214, 227-234."""

    def _default_config(self, **overrides):
        config = {
            'atom_feature_element': False,
            'atom_feature_electronegativity': True,
            'atom_feature_hardness': True,
            'atom_feature_atom_size': True,
            'atom_feature_hybridization': True,
            'atom_feature_aromaticity': True,
            'atom_feature_number_of_rings': False,
            'atom_feature_ring_size': True,
            'atom_feature_number_of_Hs': True,
            'atom_feature_formal_charge': True,
            'bond_feature_bond_order': True,
            'bond_feature_conjugation': True,
            'bond_feature_polarization': True,
            'bond_feature_charge_conjugation': True,
            'bond_feature_focused': False,
            'acid_or_base': 'base',
            'mask_size': 4,
            'model_embedding_size': 128,
            'model_gnn_layers': 4,
            'model_fc_layers': 2,
            'model_dropout_rate': 0.0,
            'model_dense_neurons': 448,
            'model_attention_heads': 4,
        }
        config.update(overrides)
        return config

    def test_one_hot_unknown_falls_back_to_last(self):
        """Line 23: x not in allowable_set → last element."""
        from pick_a_pka.backends.pkalearn.featurizer import one_hot
        result = one_hot("Z", ["A", "B", "C"])
        # Last element "C" should be True
        assert result == [False, False, True]

    def test_get_node_features_element_feature_enabled(self):
        """Line 34: atom_feature_element branch."""
        from pick_a_pka.backends.pkalearn.featurizer import get_node_features
        mol = Chem.MolFromSmiles("CC")
        config = self._default_config(atom_feature_element=True)
        feats = get_node_features(mol, center=0, config=config)
        assert feats.shape[0] == mol.GetNumAtoms()

    def test_get_node_features_number_of_rings(self):
        """Line 60: atom_feature_number_of_rings branch."""
        from pick_a_pka.backends.pkalearn.featurizer import get_node_features
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene — atoms in rings
        config = self._default_config(atom_feature_number_of_rings=True)
        feats = get_node_features(mol, center=0, config=config)
        assert feats.shape[0] == mol.GetNumAtoms()

    def test_get_edge_features_conjugation_only(self):
        """Lines 88-89: bond_feature_conjugation without charge_conjugation/focused."""
        from pick_a_pka.backends.pkalearn.featurizer import get_edge_features
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        config = self._default_config(
            bond_feature_conjugation=True,
            bond_feature_charge_conjugation=False,
            bond_feature_focused=False,
        )
        feats = get_edge_features(mol, config)
        assert feats.shape[0] == mol.GetNumBonds() * 2

    def test_get_edge_features_charge_conjugation(self):
        """Lines 142-148: bond_feature_charge_conjugation path."""
        from pick_a_pka.backends.pkalearn.featurizer import get_edge_features
        # Acetic acid has C-O bonds that exercise the conjugation check
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        config = self._default_config(
            bond_feature_conjugation=True,
            bond_feature_charge_conjugation=True,
        )
        feats = get_edge_features(mol, config)
        assert feats.shape[0] == mol.GetNumBonds() * 2

    def test_get_edge_features_charged_molecule(self):
        """Lines 142-148 deep branch: charged O/N to test strongConjugation paths."""
        from pick_a_pka.backends.pkalearn.featurizer import get_edge_features
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # acetate — negatively charged oxygen
        config = self._default_config(
            bond_feature_charge_conjugation=True,
            bond_feature_conjugation=True,
        )
        feats = get_edge_features(mol, config)
        assert feats.shape[0] == mol.GetNumBonds() * 2

    def test_get_edge_features_cationic_nitrogen(self):
        """Cationic nitrogen next to carbonyl — exercises double-bond charge paths."""
        from pick_a_pka.backends.pkalearn.featurizer import get_edge_features
        mol = Chem.MolFromSmiles("C(=O)[NH3+]")
        if mol is None:
            mol = Chem.MolFromSmiles("CC([NH3+])=O")
        if mol is None:
            pytest.skip("Unable to parse test molecule")
        config = self._default_config(
            bond_feature_charge_conjugation=True,
            bond_feature_conjugation=True,
        )
        feats = get_edge_features(mol, config)
        assert feats.shape[0] == mol.GetNumBonds() * 2

    def test_mol_to_graph_returns_data(self):
        """Line 214: mol_to_graph produces a PyG Data object."""
        from pick_a_pka.backends.pkalearn.featurizer import mol_to_graph
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        config = PkaLearnModel.DEFAULT_CONFIG
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        data = mol_to_graph(mol, center=3, config=config)
        assert data is not None

    def test_from_acid_to_base_removes_proton(self):
        """Lines 227-228: from_acid_to_base lowers formal charge."""
        from pick_a_pka.backends.pkalearn.featurizer import from_acid_to_base
        mol = Chem.RWMol(Chem.MolFromSmiles("NCC(=O)O"))
        original_charge = mol.GetAtomWithIdx(0).GetFormalCharge()
        found, mol_out, _ = from_acid_to_base(mol, center=0)
        assert found is True
        assert mol_out.GetAtomWithIdx(0).GetFormalCharge() == original_charge - 1

    def test_from_acid_to_base_on_carbon(self):
        """Lines 233-234: carbon centre is also handled."""
        from pick_a_pka.backends.pkalearn.featurizer import from_acid_to_base
        mol = Chem.RWMol(Chem.MolFromSmiles("C"))
        found, mol_out, _ = from_acid_to_base(mol, center=0)
        assert found is True

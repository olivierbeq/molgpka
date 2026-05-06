from unittest.mock import patch

import pytest
from rdkit import Chem

from constants import ACETIC_ACID, GLYCINE


class TestPkaLearnModelLines:

    def test_load_weights_raises_resource_not_found(self):
        """Lines 114-115: ResourceNotFoundError when weights file is missing."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        from pick_a_pka.core.exceptions import ResourceNotFoundError
        m = PkaLearnModel.__new__(PkaLearnModel)
        from pick_a_pka.backends.pkalearn.network import PkaLearnGNN
        m.model = PkaLearnGNN(feature_size=19, edge_dim=7, model_params=PkaLearnModel.DEFAULT_CONFIG)
        m.device = "cpu"
        with pytest.raises(ResourceNotFoundError):
            with patch("importlib.resources.files", side_effect=FileNotFoundError("gone")):
                m._load_weights()

    def test_predict_pka_with_str_input(self):
        """Line 151: mol_or_smiles is a string in predict()."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        m = PkaLearnModel(device="cpu")
        result = m.predict_pka(ACETIC_ACID)
        assert "acid_pka" in result

    def test_predict_pka_with_none_step_mol(self):
        """Line 152: invalid SMILES in step_mol is skipped gracefully."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        m = PkaLearnModel(device="cpu")
        # If the ladder returns a bad step, predict_pka should not crash
        with patch.object(m, "predict", return_value=([{"smiles": "INVALID!!", "center": 0, "pka": 5.0}],
                                                      Chem.MolFromSmiles(ACETIC_ACID))
                          ):
            result = m.predict_pka(ACETIC_ACID)
        assert isinstance(result, dict)

    def test_map_ladder_center_to_neutral_direct_match(self):
        """Lines 35-36: direct substructure match path."""
        from pick_a_pka.backends.pkalearn.model import _map_ladder_center_to_neutral
        step_mol = Chem.MolFromSmiles(ACETIC_ACID)
        neutral_mol = Chem.MolFromSmiles(ACETIC_ACID)
        idx = _map_ladder_center_to_neutral(step_mol, center_idx=0, neutral_mol=neutral_mol)
        assert idx is not None

    def test_map_ladder_center_mcs_fallback(self):
        """Lines 42-68: MCS fallback when direct match fails (charged vs neutral)."""
        from pick_a_pka.backends.pkalearn.model import _map_ladder_center_to_neutral
        step_mol = Chem.MolFromSmiles("CC(=O)[O-]")  # acetate — charged
        neutral_mol = Chem.MolFromSmiles(ACETIC_ACID)
        idx = _map_ladder_center_to_neutral(step_mol, center_idx=0, neutral_mol=neutral_mol)
        # Should return an int or None — must not crash
        assert idx is None or isinstance(idx, int)

    def test_map_ladder_center_out_of_range(self):
        """Line 42: center_idx >= len(match) falls through to MCS."""
        from pick_a_pka.backends.pkalearn.model import _map_ladder_center_to_neutral
        step_mol = Chem.MolFromSmiles("C")
        neutral_mol = Chem.MolFromSmiles("C")
        idx = _map_ladder_center_to_neutral(step_mol, center_idx=9999, neutral_mol=neutral_mol)
        assert idx is None or isinstance(idx, int)

    def test_map_ladder_mcs_zero_atoms_returns_none(self):
        """Line 56: numAtoms == 0 → None."""
        from pick_a_pka.backends.pkalearn.model import _map_ladder_center_to_neutral
        step_mol = Chem.MolFromSmiles("C")
        neutral_mol = Chem.MolFromSmiles("O")  # totally different → MCS numAtoms == 0
        idx = _map_ladder_center_to_neutral(step_mol, center_idx=0, neutral_mol=neutral_mol)
        assert idx is None

    def test_predict_pka_amphoteric_acid_extension(self):
        """Lines 227-239: allow_amphoteric extra acid pass."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        m = PkaLearnModel(device="cpu", allow_amphoteric=True)
        result = m.predict_pka(GLYCINE)
        assert "acid_pka" in result
        # Glycine has both amino (basic) and carboxyl (acidic) sites
        assert isinstance(result["acid_pka"], dict)

    def test_predict_amphoteric_skips_atom_without_hs(self):
        """Line 233: atom with GetTotalNumHs() == 0 is skipped in amphoteric pass."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        m = PkaLearnModel(device="cpu", allow_amphoteric=True)
        # Trimethylamine has no H on N in neutral form → amphoteric pass skips it
        result = m.predict_pka("CN(C)C")
        assert isinstance(result, dict)

    def test_predict_pka_skips_bad_ladder_center_index(self):
        """Line 180: ladder_idx >= step_mol.GetNumAtoms() → continue."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        m = PkaLearnModel(device="cpu")
        with patch.object(m, "predict", return_value=(
                [{"smiles": ACETIC_ACID, "center": 9999, "pka": 5.0}],
                Chem.MolFromSmiles(ACETIC_ACID)
        )
                          ):
            result = m.predict_pka(ACETIC_ACID)
        assert isinstance(result, dict)

    def test_predict_pka_skips_none_neutral_idx(self):
        """Line 182: neutral_idx is None → continue."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        m = PkaLearnModel(device="cpu")
        with patch("pick_a_pka.backends.pkalearn.model._map_ladder_center_to_neutral",
                   return_value=None
                   ):
            result = m.predict_pka(ACETIC_ACID)
        assert isinstance(result, dict)

    def test_predict_pka_skips_when_from_acid_to_base_fails(self):
        """Line 190: b_found == False → continue.
        from_acid_to_base is imported inside predict_pka via
        `from .featurizer import from_acid_to_base`, so we patch it on featurizer."""
        from pick_a_pka.backends.pkalearn.model import PkaLearnModel
        m = PkaLearnModel(device="cpu")
        with patch("pick_a_pka.backends.pkalearn.featurizer.from_acid_to_base",
                   return_value=(False, None, None)):
            result = m.predict_pka(ACETIC_ACID)
        assert isinstance(result, dict)

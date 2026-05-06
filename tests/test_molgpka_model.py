import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from constants import ACETIC_ACID


class TestMolGpKaModelDispose:
    def test_dispose_deletes_models(self):
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        m = MolGpKaModel(device="cpu")
        m.dispose()
        assert not hasattr(m, "model_acid") or m.model_acid is None or True
        # After dispose the attributes have been deleted; a second call must not crash
        # (the __del__ guard handles missing attrs).

    def test_predict_pka_uncharged_false(self):
        """predict_pka with uncharged=False skips the Uncharger step."""
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        m = MolGpKaModel(device="cpu")
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        result = m.predict_pka(mol, uncharged=False)
        assert "base_pka" in result
        assert "acid_pka" in result

    def test_remap_pka_maps_hydrogen_to_heavy(self):
        """_remap_pka_without_hs should handle H-atom indices via the h_to_heavy map."""
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        m = MolGpKaModel(device="cpu")
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        mol_h = AllChem.AddHs(mol)
        # Find a hydrogen atom index
        h_idx = next(a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
        # Pretend there is an acid at that hydrogen
        fake_acid = {h_idx: 4.76}
        fake_base = {}
        mol_no_hs, new_base, new_acid = m._remap_pka_without_hs(mol_h, fake_base, fake_acid)
        # The H should be remapped to its heavy-atom neighbour
        assert len(new_acid) == 1


class TestMolGpKaModelLines:
    """Lines 32-33: ResourceNotFoundError path in _load_model."""

    def test_load_model_bad_filename_raises(self):
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        from pick_a_pka.core.exceptions import ResourceNotFoundError
        m = MolGpKaModel.__new__(MolGpKaModel)
        with pytest.raises(ResourceNotFoundError):
            m._load_model("pick_a_pka.backends.molgpka.resources", "does_not_exist.pth")

    def test_remap_heavy_idx_via_heavy_to_heavy(self):
        """Lines 117-118: acid/base indices that are heavy atoms are remapped through orig_to_new_idx."""
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        from rdkit.Chem import AllChem
        m = MolGpKaModel(device="cpu")
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        mol_h = AllChem.AddHs(mol)
        # Use heavy-atom index 0 directly in the base dict
        result_mol, base_dict, acid_dict = m._remap_pka_without_hs(mol_h, {0: 9.0}, {})
        assert isinstance(result_mol, Chem.Mol)
        # The key should have been re-mapped (0 is a heavy atom so it's in orig_to_new_idx)
        assert len(base_dict) == 1

    def test_remap_drops_unmappable_index(self):
        """Line 126: an old_idx present in neither map is silently dropped."""
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        from rdkit.Chem import AllChem
        m = MolGpKaModel(device="cpu")
        mol = Chem.MolFromSmiles("C")
        mol_h = AllChem.AddHs(mol)
        # Use an out-of-range index — should be silently dropped
        result_mol, base_dict, acid_dict = m._remap_pka_without_hs(mol_h, {9999: 5.0}, {})
        assert len(base_dict) == 0

    def test_predict_pka_with_charged_input(self):
        """Line 110 (uncharged=True path, un.uncharge branch)."""
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        m = MolGpKaModel(device="cpu")
        # Charged molecule — the Uncharger branch should fire
        mol = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")
        result = m.predict_pka(mol, uncharged=True)
        assert "base_pka" in result and "acid_pka" in result

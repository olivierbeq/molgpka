import pytest
from rdkit import Chem

from constants import ACETIC_ACID, GLYCINE, BUTANE, ANILINE


class TestBasePKaModelStubs:
    """Test the stub methods on BasePKaModel that defer or raise."""

    def test_predict_microstates_not_implemented(self):
        """Calling predict_microstates on the base class must raise NotImplementedError."""
        from pick_a_pka.core.base import BasePKaModel
        from rdkit import Chem

        class _Concrete(BasePKaModel):
            def predict_pka(self, mol):
                return {}

        obj = _Concrete(device="cpu")
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        with pytest.raises(NotImplementedError):
            obj.predict_microstates(mol)

    def test_dispose_is_no_op(self):
        """dispose() on the base class must not raise."""
        from pick_a_pka.core.base import BasePKaModel

        class _Concrete(BasePKaModel):
            def predict_pka(self, mol):
                return {}

        obj = _Concrete()
        obj.dispose()  # must not raise

    def test_to_mol_invalid_type_raises(self):
        """Passing an integer to _to_mol should raise InvalidMoleculeError."""
        from pick_a_pka.core.base import BasePKaModel
        from pick_a_pka.core.exceptions import InvalidMoleculeError

        class _Concrete(BasePKaModel):
            def predict_pka(self, mol):
                return {}

        obj = _Concrete()
        with pytest.raises(InvalidMoleculeError):
            obj._to_mol(12345)

    def test_to_mol_invalid_smiles_raises(self):
        from pick_a_pka.core.base import BasePKaModel
        from pick_a_pka.core.exceptions import InvalidMoleculeError

        class _Concrete(BasePKaModel):
            def predict_pka(self, mol):
                return {}

        obj = _Concrete()
        with pytest.raises(InvalidMoleculeError):
            obj._to_mol("NOT_A_VALID_SMILES!!!")

    def test_to_mol_list_of_smiles(self):
        """_to_mol with a list of SMILES strings returns a flat list of mols."""
        from pick_a_pka.core.base import BasePKaModel

        class _Concrete(BasePKaModel):
            def predict_pka(self, mol):
                return {}

        obj = _Concrete()
        mols = obj._to_mol([ACETIC_ACID, GLYCINE])
        assert len(mols) == 2
        assert all(isinstance(m, Chem.Mol) for m in mols)


class TestPKaPredictorListInput:
    """predict_pka and predict_microstates with list inputs (returns list)."""

    @pytest.fixture(scope="class")
    def model(self):
        from pick_a_pka import PKaPredictor
        return PKaPredictor("molgpka")

    def test_predict_pka_list_returns_list(self, model):
        results = model.predict_pka([ACETIC_ACID, GLYCINE])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_microstates_list_returns_list(self, model):
        results = model.predict_microstates([ACETIC_ACID, GLYCINE], ph=7.4)
        assert isinstance(results, list)
        assert len(results) == 2


class TestProtonationLadder:
    """Tests for PKaPredictor.protonation_ladder."""

    @pytest.fixture(scope="class")
    def model(self):
        from pick_a_pka import PKaPredictor
        return PKaPredictor("molgpka")

    def test_acid_first_true(self, model):
        ladder = model.protonation_ladder(GLYCINE, acid_first=True)
        assert isinstance(ladder, list)
        assert len(ladder) >= 1

    def test_acid_first_false_reverses_order(self, model):
        fwd = model.protonation_ladder(GLYCINE, acid_first=True)
        rev = model.protonation_ladder(GLYCINE, acid_first=False)
        assert list(reversed(fwd)) == rev

    def test_non_ionisable_molecule_returns_one_entry(self, model):
        """CCCC has no ionisable sites; the ladder should still return ≥1 entry."""
        ladder = model.protonation_ladder(BUTANE)
        assert len(ladder) >= 1

    def test_pkalearn_backend_ladder(self):
        from pick_a_pka import PKaPredictor
        model = PKaPredictor("pkalearn")
        ladder = model.protonation_ladder(ACETIC_ACID, acid_first=True)
        assert isinstance(ladder, list)
        assert len(ladder) >= 1

    def test_no_duplicates_in_ladder(self, model):
        ladder = model.protonation_ladder(GLYCINE)
        assert len(ladder) == len(set(ladder))


class TestGenerateOrderedStates:
    """Unit-test _generate_ordered_states directly."""

    def _fn(self):
        from pick_a_pka.predictor import _generate_ordered_states
        return _generate_ordered_states

    def test_no_ionisable_sites_returns_single_mol(self):
        fn = self._fn()
        mol = Chem.MolFromSmiles(BUTANE)
        states = fn(mol, {}, {})
        assert len(states) == 1

    def test_one_acid_site(self):
        fn = self._fn()
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        # atom 3 is the OH oxygen (approximate)
        states = fn(mol, {}, {3: 4.76})
        assert len(states) >= 1

    def test_one_base_site(self):
        fn = self._fn()
        mol = Chem.MolFromSmiles(ANILINE)
        states = fn(mol, {6: 4.6}, {})
        assert len(states) >= 1

    def test_amphoteric_both_dicts(self):
        fn = self._fn()
        mol = Chem.MolFromSmiles(GLYCINE)
        states = fn(mol, {0: 9.6}, {4: 2.3})
        assert len(states) == 3

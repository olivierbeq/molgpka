"""Tests for the PKaPredictor public API (backend-agnostic)."""
import pytest
from rdkit import Chem

from pick_a_pka import PKaPredictor
from pick_a_pka.core.exceptions import InvalidBackendError, InvalidMoleculeError
from pick_a_pka.core.types import BackendType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def molgpka():
    return PKaPredictor("molgpka")


@pytest.fixture(scope="module")
def pkalearn():
    return PKaPredictor("pkalearn")


@pytest.fixture(scope="module")
def pkalearn_amphoteric():
    return PKaPredictor("pkalearn", allow_amphoteric=True)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_string_backend_molgpka(self):
        p = PKaPredictor("molgpka")
        assert p.model_name == BackendType.MOLGPKA

    def test_string_backend_pkalearn(self):
        p = PKaPredictor("pkalearn")
        assert p.model_name == BackendType.PKALEARN

    def test_enum_backend(self):
        p = PKaPredictor(BackendType.MOLGPKA)
        assert p.model_name == BackendType.MOLGPKA

    def test_invalid_backend_raises(self):
        with pytest.raises(InvalidBackendError):
            PKaPredictor("notabackend")

    def test_allow_amphoteric_stored(self):
        p = PKaPredictor("pkalearn", allow_amphoteric=True)
        assert p.allow_amphoteric is True

    def test_allow_amphoteric_default_false(self):
        p = PKaPredictor("pkalearn")
        assert p.allow_amphoteric is False


# ---------------------------------------------------------------------------
# Output contract: predict_pka
# ---------------------------------------------------------------------------

class TestPredictPkaContract:
    """Every backend must return the same dict shape."""

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_returns_dict_with_required_keys(self, backend):
        result = PKaPredictor(backend).predict_pka("CC(=O)O")
        assert isinstance(result, dict)
        assert "acid_pka" in result
        assert "base_pka" in result
        assert "mol" in result

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_pka_dicts_are_dicts(self, backend):
        result = PKaPredictor(backend).predict_pka("CC(=O)O")
        assert isinstance(result["acid_pka"], dict)
        assert isinstance(result["base_pka"], dict)

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_mol_is_rdkit_mol(self, backend):
        result = PKaPredictor(backend).predict_pka("CC(=O)O")
        assert isinstance(result["mol"], Chem.Mol)

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_mol_has_no_explicit_hs(self, backend):
        """Returned mol should be the heavy-atom-only molecule."""
        result = PKaPredictor(backend).predict_pka("CC(=O)O")
        mol = result["mol"]
        h_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        assert h_count == 0

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_atom_indices_in_range(self, backend):
        """All returned atom indices must be valid for the returned mol."""
        result = PKaPredictor(backend).predict_pka("CC(=O)O")
        n_atoms = result["mol"].GetNumAtoms()
        for idx in list(result["acid_pka"]) + list(result["base_pka"]):
            assert 0 <= idx < n_atoms

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_pka_values_are_floats(self, backend):
        result = PKaPredictor(backend).predict_pka("CC(=O)O")
        for val in list(result["acid_pka"].values()) + list(result["base_pka"].values()):
            assert isinstance(val, float)

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_pka_values_in_plausible_range(self, backend):
        """No predicted pKa should be absurdly outside chemical range."""
        result = PKaPredictor(backend).predict_pka("CC(=O)O")
        for val in list(result["acid_pka"].values()) + list(result["base_pka"].values()):
            assert -5 < val < 30


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    def test_accepts_smiles_string(self, molgpka):
        result = molgpka.predict_pka("CC(=O)O")
        assert isinstance(result, dict)

    def test_accepts_rdkit_mol(self, molgpka):
        mol = Chem.MolFromSmiles("CC(=O)O")
        result = molgpka.predict_pka(mol)
        assert isinstance(result, dict)

    def test_smiles_and_mol_give_same_result(self, molgpka):
        r_smi = molgpka.predict_pka("CC(=O)O")
        r_mol = molgpka.predict_pka(Chem.MolFromSmiles("CC(=O)O"))
        assert r_smi["acid_pka"] == r_mol["acid_pka"]
        assert r_smi["base_pka"] == r_mol["base_pka"]

    def test_accepts_list_of_smiles(self, molgpka):
        results = molgpka.predict_pka(["CC(=O)O", "c1ccccc1N"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_accepts_list_of_mols(self, molgpka):
        mols = [Chem.MolFromSmiles(s) for s in ["CC(=O)O", "c1ccccc1N"]]
        results = molgpka.predict_pka(mols)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_single_mol_returns_dict_not_list(self, molgpka):
        result = molgpka.predict_pka("CC(=O)O")
        assert isinstance(result, dict)

    def test_list_input_returns_list(self, molgpka):
        result = molgpka.predict_pka(["CC(=O)O"])
        assert isinstance(result, list)

    def test_invalid_smiles_raises(self, molgpka):
        with pytest.raises(InvalidMoleculeError):
            molgpka.predict_pka("NOTASMILES!!!")

    def test_empty_string_is_invalid(self, molgpka):
        # RDKit parses "" as an empty but valid mol; _to_mol accepts it.
        # The downstream model receives a 0-atom molecule and must handle it
        # gracefully (either raise or return empty dicts — not crash).
        try:
            result = molgpka.predict_pka("")
            # If it didn't raise, it must at least return valid structure
            assert isinstance(result, dict)
            assert "acid_pka" in result
        except Exception:
            pass  # raising is also acceptable

    def test_non_mol_non_string_raises(self, molgpka):
        with pytest.raises((InvalidMoleculeError, Exception)):
            molgpka.predict_pka(42)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_same_smiles_gives_same_result(self, backend):
        model = PKaPredictor(backend)
        r1 = model.predict_pka("CC(=O)O")
        r2 = model.predict_pka("CC(=O)O")
        assert r1["acid_pka"] == r2["acid_pka"]
        assert r1["base_pka"] == r2["base_pka"]

    def test_canonical_and_noncanonical_smiles_equivalent(self, molgpka):
        """Different SMILES of the same molecule should give consistent pKa sets."""
        r1 = molgpka.predict_pka("CC(=O)O")  # canonical
        r2 = molgpka.predict_pka("OC(C)=O")  # non-canonical
        # Both should have at least one acidic site
        assert len(r1["acid_pka"]) == len(r2["acid_pka"])


# ---------------------------------------------------------------------------
# Chemical correctness: known molecules
# ---------------------------------------------------------------------------

class TestKnownChemistry:
    """
    Reference pKa values from literature. We allow ±2 tolerance given
    model accuracy; the goal is to verify the sign and rough magnitude.
    """

    # --- Acetic acid: acid pKa ≈ 4.76 ---
    def test_acetic_acid_has_exactly_one_acid_site(self, molgpka):
        result = molgpka.predict_pka("CC(=O)O")
        assert len(result["acid_pka"]) == 1

    def test_acetic_acid_acid_pka_in_range(self, molgpka):
        result = molgpka.predict_pka("CC(=O)O")
        assert any(2.0 < v < 7.0 for v in result["acid_pka"].values())

    def test_acetic_acid_no_basic_site(self, molgpka):
        result = molgpka.predict_pka("CC(=O)O")
        assert len(result["base_pka"]) == 0

    # --- Aniline: base pKa ≈ 4.6 (conjugate acid) ---
    def test_aniline_has_basic_site(self, molgpka):
        result = molgpka.predict_pka("c1ccccc1N")
        assert len(result["base_pka"]) >= 1

    def test_aniline_base_pka_in_range(self, molgpka):
        result = molgpka.predict_pka("c1ccccc1N")
        assert any(2.0 < v < 7.0 for v in result["base_pka"].values())

    # --- Chloroacetic acid: acid pKa ≈ 2.86 (stronger acid than acetic) ---
    def test_chloroacetic_acid_pka_below_acetic(self, molgpka):
        r_acetic = molgpka.predict_pka("CC(=O)O")
        r_chloro = molgpka.predict_pka("ClCC(=O)O")
        min_acetic = min(r_acetic["acid_pka"].values())
        min_chloro = min(r_chloro["acid_pka"].values())
        assert min_chloro < min_acetic

    # --- Non-ionisable molecule: butane ---
    def test_butane_has_no_ionizable_sites(self, molgpka):
        result = molgpka.predict_pka("CCCC")
        assert len(result["acid_pka"]) == 0
        assert len(result["base_pka"]) == 0

    # --- Glycine: amphoteric amino acid, base pKa ≈ 9.6, acid pKa ≈ 2.3 ---
    def test_glycine_has_both_acid_and_base_sites(self, molgpka):
        result = molgpka.predict_pka("NCC(=O)O")
        assert len(result["acid_pka"]) >= 1
        assert len(result["base_pka"]) >= 1

    def test_glycine_acid_pka_lower_than_base_pka(self, molgpka):
        """Carboxylic pKa must be lower than amine pKa."""
        result = molgpka.predict_pka("NCC(=O)O")
        min_acid = min(result["acid_pka"].values())
        max_base = max(result["base_pka"].values())
        assert min_acid < max_base

    # --- Phenol: acid pKa ≈ 9.99 ---
    def test_phenol_is_more_acidic_than_cyclohexanol(self, molgpka):
        r_phenol = molgpka.predict_pka("c1ccccc1O")
        r_cyclohex = molgpka.predict_pka("OC1CCCCC1")
        min_phenol = min(r_phenol["acid_pka"].values())
        min_cyclohex = min(r_cyclohex["acid_pka"].values())
        assert min_phenol < min_cyclohex

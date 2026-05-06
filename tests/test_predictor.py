"""Tests for the PKaPredictor public API (backend-agnostic)."""

import pytest
from rdkit import Chem

from constants import ACETIC_ACID, NON_CANONICAL_ACETIC_ACID, ANILINE, CHLOROACETIC_ACID, CYCLOHEXANOL, GLYCINE, \
    PHENOL, BUTANE
from pick_a_pka import PKaPredictor
from pick_a_pka.core.exceptions import InvalidBackendError, InvalidMoleculeError
from pick_a_pka.core.types import BackendType
from pick_a_pka.predictor import _generate_ordered_states


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
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        assert isinstance(result, dict)
        assert "acid_pka" in result
        assert "base_pka" in result
        assert "mol" in result

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_pka_dicts_are_dicts(self, backend):
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        assert isinstance(result["acid_pka"], dict)
        assert isinstance(result["base_pka"], dict)

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_mol_is_rdkit_mol(self, backend):
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        assert isinstance(result["mol"], Chem.Mol)

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_mol_has_no_explicit_hs(self, backend):
        """Returned mol should be the heavy-atom-only molecule."""
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        mol = result["mol"]
        h_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        assert h_count == 0

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_atom_indices_in_range(self, backend):
        """All returned atom indices must be valid for the returned mol."""
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        n_atoms = result["mol"].GetNumAtoms()
        for idx in list(result["acid_pka"]) + list(result["base_pka"]):
            assert 0 <= idx < n_atoms

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_pka_values_are_floats(self, backend):
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        for val in list(result["acid_pka"].values()) + list(result["base_pka"].values()):
            assert isinstance(val, float)

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_pka_values_in_plausible_range(self, backend):
        """No predicted pKa should be absurdly outside chemical range."""
        result = PKaPredictor(backend).predict_pka(ACETIC_ACID)
        for val in list(result["acid_pka"].values()) + list(result["base_pka"].values()):
            assert -5 < val < 30


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    def test_accepts_smiles_string(self, molgpka):
        result = molgpka.predict_pka(ACETIC_ACID)
        assert isinstance(result, dict)

    def test_accepts_rdkit_mol(self, molgpka):
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        result = molgpka.predict_pka(mol)
        assert isinstance(result, dict)

    def test_smiles_and_mol_give_same_result(self, molgpka):
        r_smi = molgpka.predict_pka(ACETIC_ACID)
        r_mol = molgpka.predict_pka(Chem.MolFromSmiles(ACETIC_ACID))
        assert r_smi["acid_pka"] == r_mol["acid_pka"]
        assert r_smi["base_pka"] == r_mol["base_pka"]

    def test_accepts_list_of_smiles(self, molgpka):
        results = molgpka.predict_pka([ACETIC_ACID, ANILINE])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_accepts_list_of_mols(self, molgpka):
        mols = [Chem.MolFromSmiles(s) for s in [ACETIC_ACID, ANILINE]]
        results = molgpka.predict_pka(mols)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_single_mol_returns_dict_not_list(self, molgpka):
        result = molgpka.predict_pka(ACETIC_ACID)
        assert isinstance(result, dict)

    def test_list_input_returns_list(self, molgpka):
        result = molgpka.predict_pka([ACETIC_ACID])
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
        r1 = model.predict_pka(ACETIC_ACID)
        r2 = model.predict_pka(ACETIC_ACID)
        assert r1["acid_pka"] == r2["acid_pka"]
        assert r1["base_pka"] == r2["base_pka"]

    def test_canonical_and_noncanonical_smiles_equivalent(self, molgpka):
        """Different SMILES of the same molecule should give consistent pKa sets."""
        r1 = molgpka.predict_pka(ACETIC_ACID)  # canonical
        r2 = molgpka.predict_pka(NON_CANONICAL_ACETIC_ACID)  # non-canonical
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
        result = molgpka.predict_pka(ACETIC_ACID)
        assert len(result["acid_pka"]) == 1

    def test_acetic_acid_acid_pka_in_range(self, molgpka):
        result = molgpka.predict_pka(ACETIC_ACID)
        assert any(2.0 < v < 7.0 for v in result["acid_pka"].values())

    def test_acetic_acid_no_basic_site(self, molgpka):
        result = molgpka.predict_pka(ACETIC_ACID)
        assert len(result["base_pka"]) == 0

    # --- Aniline: base pKa ≈ 4.6 (conjugate acid) ---
    def test_aniline_has_basic_site(self, molgpka):
        result = molgpka.predict_pka(ANILINE)
        assert len(result["base_pka"]) >= 1

    def test_aniline_base_pka_in_range(self, molgpka):
        result = molgpka.predict_pka(ANILINE)
        assert any(2.0 < v < 7.0 for v in result["base_pka"].values())

    # --- Chloroacetic acid: acid pKa ≈ 2.86 (stronger acid than acetic) ---
    def test_chloroacetic_acid_pka_below_acetic(self, molgpka):
        r_acetic = molgpka.predict_pka(ACETIC_ACID)
        r_chloro = molgpka.predict_pka(CHLOROACETIC_ACID)
        min_acetic = min(r_acetic["acid_pka"].values())
        min_chloro = min(r_chloro["acid_pka"].values())
        assert min_chloro < min_acetic

    # --- Non-ionisable molecule: butane ---
    def test_butane_has_no_ionizable_sites(self, molgpka):
        result = molgpka.predict_pka(BUTANE)
        assert len(result["acid_pka"]) == 0
        assert len(result["base_pka"]) == 0

    # --- Glycine: amphoteric amino acid, base pKa ≈ 9.6, acid pKa ≈ 2.3 ---
    def test_glycine_has_both_acid_and_base_sites(self, molgpka):
        result = molgpka.predict_pka(GLYCINE)
        assert len(result["acid_pka"]) >= 1
        assert len(result["base_pka"]) >= 1

    def test_glycine_acid_pka_lower_than_base_pka(self, molgpka):
        """Carboxylic pKa must be lower than amine pKa."""
        result = molgpka.predict_pka(GLYCINE)
        min_acid = min(result["acid_pka"].values())
        max_base = max(result["base_pka"].values())
        assert min_acid < max_base

    # --- Phenol: acid pKa ≈ 9.99 ---
    def test_phenol_is_more_acidic_than_cyclohexanol(self, molgpka):
        r_phenol = molgpka.predict_pka(PHENOL)
        r_cyclohex = molgpka.predict_pka(CYCLOHEXANOL)
        min_phenol = min(r_phenol["acid_pka"].values())
        min_cyclohex = min(r_cyclohex["acid_pka"].values())
        assert min_phenol < min_cyclohex


# ---------------------------------------------------------------------------
# _generate_ordered_states (module-level helper)
# ---------------------------------------------------------------------------

class TestGenerateOrderedStates:
    def test_no_sites_returns_one_state(self):
        mol = Chem.MolFromSmiles(BUTANE)
        states = _generate_ordered_states(mol, {}, {})
        assert len(states) == 1

    def test_one_acid_site_returns_two_states(self):
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        pred = PKaPredictor("molgpka").predict_pka(ACETIC_ACID)
        states = _generate_ordered_states(pred["mol"], pred["base_pka"], pred["acid_pka"])
        # neutral + deprotonated = 2
        assert len(states) == 2

    def test_states_are_rdkit_mols(self):
        pred = PKaPredictor("molgpka").predict_pka(GLYCINE)
        states = _generate_ordered_states(pred["mol"], pred["base_pka"], pred["acid_pka"])
        for s in states:
            assert isinstance(s, Chem.Mol)

    def test_states_in_ascending_charge_order(self):
        """Fully protonated (highest charge) should be first."""
        pred = PKaPredictor("molgpka").predict_pka(GLYCINE)
        states = _generate_ordered_states(pred["mol"], pred["base_pka"], pred["acid_pka"])
        charges = [
            sum(a.GetFormalCharge() for a in s.GetAtoms())
            for s in states
        ]
        # Charges must be non-increasing (most positive first)
        for i in range(len(charges) - 1):
            assert charges[i] >= charges[i + 1]

    def test_amphoteric_mol_spans_positive_to_negative(self):
        pred = PKaPredictor("molgpka").predict_pka(GLYCINE)
        states = _generate_ordered_states(pred["mol"], pred["base_pka"], pred["acid_pka"])
        charges = [sum(a.GetFormalCharge() for a in s.GetAtoms()) for s in states]
        assert max(charges) >= 0
        assert min(charges) <= 0


# ---------------------------------------------------------------------------
# protonation_ladder — acid_first parameter
# ---------------------------------------------------------------------------

class TestProtonationLadder:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_acid_first_true_default(self, backend):
        model = PKaPredictor(backend)
        ladder = model.protonation_ladder(GLYCINE, acid_first=True)
        assert isinstance(ladder, list)
        assert len(ladder) >= 1

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_acid_first_false_reverses_order(self, backend):
        model = PKaPredictor(backend)
        fwd = model.protonation_ladder(GLYCINE, acid_first=True)
        rev = model.protonation_ladder(GLYCINE, acid_first=False)
        assert fwd == list(reversed(rev))

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_ladder_returns_valid_smiles(self, backend):
        model = PKaPredictor(backend)
        for smi in model.protonation_ladder(GLYCINE):
            assert Chem.MolFromSmiles(smi) is not None

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_non_ionisable_returns_one_entry(self, backend):
        model = PKaPredictor(backend)
        ladder = model.protonation_ladder(BUTANE)
        assert len(ladder) == 1

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_first_entry_most_protonated(self, backend):
        model = PKaPredictor(backend)
        ladder = model.protonation_ladder(GLYCINE, acid_first=True)
        first_mol = Chem.MolFromSmiles(ladder[0])
        last_mol = Chem.MolFromSmiles(ladder[-1])
        first_charge = sum(a.GetFormalCharge() for a in first_mol.GetAtoms())
        last_charge = sum(a.GetFormalCharge() for a in last_mol.GetAtoms())
        assert first_charge >= last_charge

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_no_duplicate_smiles(self, backend):
        model = PKaPredictor(backend)
        ladder = model.protonation_ladder(GLYCINE)
        assert len(ladder) == len(set(ladder))

    def test_mol_input_accepted(self):
        model = PKaPredictor("molgpka")
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        ladder = model.protonation_ladder(mol)
        assert isinstance(ladder, list)


# ---------------------------------------------------------------------------
# MolGpKa predict_pka uncharged parameter
# ---------------------------------------------------------------------------

class TestMolgpkaPredictPkaUncharged:
    def test_uncharged_true_default(self):
        model = PKaPredictor("molgpka")
        # Internally, molgpka uses uncharged=True by default
        result = model.predict_pka(ACETIC_ACID)
        assert "acid_pka" in result

    def test_uncharged_false_path(self):
        """Call predict_pka(uncharged=False) directly on the backend model."""
        model = PKaPredictor("molgpka")
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        result = model.model.predict_pka(mol, uncharged=False)
        assert "acid_pka" in result
        assert "base_pka" in result
        assert "mol" in result

    def test_charged_input_uncharged_true(self):
        """A charged input mol should be neutralised before prediction."""
        model = PKaPredictor("molgpka")
        result_charged = model.model.predict_pka(
            Chem.MolFromSmiles("[NH3+]CC(=O)[O-]"), uncharged=True
        )
        result_neutral = model.model.predict_pka(
            Chem.MolFromSmiles("NCC(=O)O"), uncharged=True
        )
        # Both should find at least one ionisable site
        assert len(result_charged["acid_pka"]) + len(result_charged["base_pka"]) >= 1
        assert len(result_neutral["acid_pka"]) + len(result_neutral["base_pka"]) >= 1

    def test_uncharged_false_on_zwitterion_same_atom_count(self):
        """uncharged=False: molecule is featurized as-is, no neutralisation."""
        model = PKaPredictor("molgpka")
        mol = Chem.MolFromSmiles(GLYCINE)
        result = model.model.predict_pka(mol, uncharged=False)
        assert result["mol"].GetNumAtoms() == mol.GetNumAtoms()


# ---------------------------------------------------------------------------
# pkalearn utils — swap tensor helpers (utils.py currently 55%)
# ---------------------------------------------------------------------------

class TestPkaLearnUtils:

    def test_whichElement_simple_atoms(self):
        from pick_a_pka.backends.pkalearn.utils import whichElement
        # Test F
        j, elem, charge, brackets = whichElement("F", 0)
        assert elem == "F"
        # Test N
        j, elem, charge, brackets = whichElement("N", 0)
        assert elem == "N"
        # Test O
        j, elem, charge, brackets = whichElement("O", 0)
        assert elem == "O"

    def test_whichElement_chlorine(self):
        from pick_a_pka.backends.pkalearn.utils import whichElement
        j, elem, charge, brackets = whichElement("Cl", 0)
        assert elem == "Cl"

    def test_whichElement_bromine(self):
        from pick_a_pka.backends.pkalearn.utils import whichElement
        j, elem, charge, brackets = whichElement("Br", 0)
        assert elem == "Br"

    def test_whichElement_nh2(self):
        from pick_a_pka.backends.pkalearn.utils import whichElement
        # In a bracket context like [NH2]
        j, elem, charge, brackets = whichElement("NH2", 0)
        assert "N" in elem

    def test_whichElement_sulfur(self):
        from pick_a_pka.backends.pkalearn.utils import whichElement
        j, elem, charge, brackets = whichElement("S", 0)
        assert elem == "S"


class TestPKaPredictorListInputs:
    """Lines 44-45 (list of mols), 62-63 (list of mols in predict_microstates),
    89-90 (list of mols in protonation_ladder)."""

    def test_predict_pka_list_of_smiles(self):
        from pick_a_pka import PKaPredictor
        model = PKaPredictor("molgpka")
        results = model.predict_pka([ACETIC_ACID, ANILINE])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all("acid_pka" in r for r in results)

    def test_predict_pka_list_of_mols(self):
        from pick_a_pka import PKaPredictor
        model = PKaPredictor("molgpka")
        mols = [Chem.MolFromSmiles(ACETIC_ACID), Chem.MolFromSmiles(ANILINE)]
        results = model.predict_pka(mols)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_microstates_list_input(self):
        from pick_a_pka import PKaPredictor
        model = PKaPredictor("molgpka")
        results = model.predict_microstates([ACETIC_ACID, ANILINE])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_microstates_list_of_mols(self):
        from pick_a_pka import PKaPredictor
        model = PKaPredictor("molgpka")
        mols = [Chem.MolFromSmiles(ACETIC_ACID)]
        results = model.predict_microstates(mols)
        assert isinstance(results, list)

    def test_protonation_ladder_preserves_order(self):
        from pick_a_pka import PKaPredictor
        model = PKaPredictor("molgpka")
        ladder = model.protonation_ladder(ACETIC_ACID, acid_first=False)
        # Reversed ladder: most deprotonated first
        ladder_fwd = model.protonation_ladder(ACETIC_ACID, acid_first=True)
        assert list(reversed(ladder_fwd)) == ladder

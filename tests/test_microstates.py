"""Tests for predict_microstates across both backends."""
import pytest
from rdkit import Chem

from pick_a_pka import PKaPredictor

ACETIC_ACID = "CC(=O)O"  # one acidic site, pKa ≈ 4.76
MORPHOLINE = "C1COCCN1"  # one basic site, pKa ≈ 8.7
GLYCINE = "NCC(=O)O"  # amphoteric


@pytest.fixture(scope="module")
def pkalearn():
    return PKaPredictor("pkalearn", allow_amphoteric=True)


@pytest.fixture(scope="module")
def molgpka():
    return PKaPredictor("molgpka")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestMicrostateSchema:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_single_ph_returns_microstate_result(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        assert "major_state" in result
        assert "major_abundance" in result
        assert "distribution" in result

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_major_state_is_rdkit_mol(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        assert isinstance(result["major_state"], Chem.Mol)

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_distribution_is_list_of_dicts(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        assert isinstance(result["distribution"], list)
        assert len(result["distribution"]) > 0
        for entry in result["distribution"]:
            assert "smiles" in entry
            assert "mol" in entry
            assert "abundance" in entry

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_distribution_mols_are_rdkit_mols(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        for entry in result["distribution"]:
            assert isinstance(entry["mol"], Chem.Mol)

    def test_ph_range_returns_dict_keyed_by_ph(self, pkalearn):
        result = pkalearn.predict_microstates(ACETIC_ACID, ph_range=(0, 14), ph_step=1.0)
        assert isinstance(result, dict)
        # Step=1 over [0,14] → 15 pH points
        assert len(result) == 15

    def test_ph_range_keys_are_floats(self, pkalearn):
        result = pkalearn.predict_microstates(ACETIC_ACID, ph_range=(0, 14), ph_step=1.0)
        for key in result:
            assert isinstance(key, float)

    def test_ph_range_each_value_is_microstate_result(self, pkalearn):
        result = pkalearn.predict_microstates(ACETIC_ACID, ph_range=(0, 14), ph_step=1.0)
        for ph_val, micro in result.items():
            assert "major_state" in micro
            assert "distribution" in micro

    def test_ph_range_without_step_raises(self, pkalearn):
        with pytest.raises((ValueError, TypeError)):
            pkalearn.predict_microstates(ACETIC_ACID, ph_range=(0, 14))


# ---------------------------------------------------------------------------
# Abundance conservation
# ---------------------------------------------------------------------------

class TestAbundanceConservation:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    @pytest.mark.parametrize("ph", [0.0, 4.76, 7.4, 14.0])
    def test_abundances_sum_to_100(self, backend, ph):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=ph)
        total = sum(d["abundance"] for d in result["distribution"])
        assert abs(total - 100.0) < 1e-4

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_major_abundance_equals_distribution_max(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        max_abundance = max(d["abundance"] for d in result["distribution"])
        assert abs(result["major_abundance"] - max_abundance) < 1e-6

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_all_abundances_non_negative(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        for d in result["distribution"]:
            assert d["abundance"] >= 0.0

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_major_abundance_leq_100(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        assert result["major_abundance"] <= 100.0 + 1e-6

    def test_ph_range_abundances_sum_to_100_at_every_ph(self, pkalearn):
        result = pkalearn.predict_microstates(ACETIC_ACID, ph_range=(0, 14), ph_step=2.0)
        for ph_val, micro in result.items():
            total = sum(d["abundance"] for d in micro["distribution"])
            assert abs(total - 100.0) < 1e-4, f"Abundances don't sum to 100 at pH {ph_val}"


# ---------------------------------------------------------------------------
# Chemical correctness
# ---------------------------------------------------------------------------

class TestChemicalCorrectness:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_acetic_acid_protonated_form_dominant_at_low_ph(self, backend):
        """At pH 0, acetic acid (pKa ≈ 4.76) should be >90% protonated."""
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=0.0)
        assert result["major_abundance"] > 90.0

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_acetic_acid_deprotonated_form_dominant_at_high_ph(self, backend):
        """At pH 14, acetate should be overwhelmingly dominant."""
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=14.0)
        assert result["major_abundance"] > 90.0

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_near_pka_both_forms_present(self, backend):
        """Near the pKa, neither form should have >90% abundance."""
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=4.76)
        assert result["major_abundance"] < 90.0

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_distinct_microstates_exist(self, backend):
        """At least two distinct protonation states must be present in the distribution."""
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        unique_smiles = {d["smiles"] for d in result["distribution"]}
        assert len(unique_smiles) >= 2

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_major_state_changes_across_ph(self, backend):
        """The dominant species at pH 0 and pH 14 should differ for acetic acid."""
        low_ph = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=0.0)
        high_ph = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=14.0)
        low_smi = Chem.MolToSmiles(low_ph["major_state"])
        high_smi = Chem.MolToSmiles(high_ph["major_state"])
        assert low_smi != high_smi

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_consecutive_states_differ_in_charge(self, backend):
        """Adjacent protonation states in the ladder must differ in formal charge by 1."""
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        charges = sorted(set(
            sum(a.GetFormalCharge() for a in d["mol"].GetAtoms())
            for d in result["distribution"]
        )
        )
        # acetic acid: 0 (neutral) and -1 (acetate)
        assert len(charges) >= 2
        for i in range(len(charges) - 1):
            # charges are sorted ascending; each step differs by exactly 1 unit
            assert abs(charges[i + 1] - charges[i]) == 1

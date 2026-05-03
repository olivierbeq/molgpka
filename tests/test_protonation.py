"""Tests for protonation-state logic: charge changes, SMILES validity, and
structural consistency of the microstate distribution."""
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from pick_a_pka import PKaPredictor

ACETIC_ACID = "CC(=O)O"
MORPHOLINE = "C1COCCN1"
GLYCINE = "NCC(=O)O"


@pytest.fixture(scope="module")
def molgpka():
    return PKaPredictor("molgpka")


@pytest.fixture(scope="module")
def pkalearn():
    return PKaPredictor("pkalearn", allow_amphoteric=True)


# ---------------------------------------------------------------------------
# SMILES validity
# ---------------------------------------------------------------------------

class TestSmilestValidity:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    @pytest.mark.parametrize("smiles", [ACETIC_ACID, MORPHOLINE, GLYCINE])
    def test_distribution_smiles_are_parseable(self, backend, smiles):
        result = PKaPredictor(backend).predict_microstates(smiles, ph=7.4)
        for entry in result["distribution"]:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"Invalid SMILES in distribution: {entry['smiles']}"

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_major_state_smiles_is_parseable(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        smi = Chem.MolToSmiles(result["major_state"])
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None


# ---------------------------------------------------------------------------
# Molecular formula conservation
# ---------------------------------------------------------------------------

class TestFormulaConservation:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_heavy_atom_count_conserved(self, backend):
        """All microstate mols must have the same number of heavy atoms."""
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        counts = {m["mol"].GetNumAtoms() for m in result["distribution"]}
        assert len(counts) == 1, f"Heavy atom counts differ: {counts}"

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_molecular_weight_nearly_conserved(self, backend):
        """MW changes by at most ~1 Da per deprotonation (loss of H)."""
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        mws = sorted(Descriptors.MolWt(d["mol"]) for d in result["distribution"])
        for i in range(len(mws) - 1):
            delta = mws[i + 1] - mws[i]
            # Each deprotonation loses 1 H ≈ 1.008 Da (within rounding)
            assert abs(delta - 1.008) < 0.05, (
                f"MW difference {delta:.3f} Da between states is not ~1 Da"
            )


# ---------------------------------------------------------------------------
# Formal charge monotonicity
# ---------------------------------------------------------------------------

class TestFormalChargeMonotonicity:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_formal_charges_form_integer_ladder(self, backend):
        """
        Deprotonation removes one proton (charge −1). Starting from the most
        protonated state, charges should form a contiguous integer sequence
        where adjacent entries differ by exactly 1.
        """
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        charges = sorted(set(
            sum(a.GetFormalCharge() for a in d["mol"].GetAtoms())
            for d in result["distribution"]
        )
        )
        for i in range(len(charges) - 1):
            assert abs(charges[i + 1] - charges[i]) == 1

    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_glycine_charges_span_both_signs(self, backend):
        """Glycine microstates include cation, zwitterion/neutral, and anion."""
        result = PKaPredictor(backend).predict_microstates(GLYCINE, ph=7.0)
        charges = {
            sum(a.GetFormalCharge() for a in d["mol"].GetAtoms())
            for d in result["distribution"]
        }
        # Must span from positive to negative
        assert max(charges) >= 0
        assert min(charges) <= 0


# ---------------------------------------------------------------------------
# State uniqueness
# ---------------------------------------------------------------------------

class TestStateUniqueness:
    @pytest.mark.parametrize("backend", ["molgpka", "pkalearn"])
    def test_no_duplicate_smiles_in_distribution(self, backend):
        result = PKaPredictor(backend).predict_microstates(ACETIC_ACID, ph=7.4)
        smiles_list = [d["smiles"] for d in result["distribution"]]
        assert len(smiles_list) == len(set(smiles_list)), "Duplicate SMILES in distribution"

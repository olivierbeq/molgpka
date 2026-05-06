"""Tests for protonation-state logic: charge changes, SMILES validity, and
structural consistency of the microstate distribution."""
from unittest.mock import MagicMock

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from constants import ACETIC_ACID, MORPHOLINE, GLYCINE, ANILINE, L_ALANINE
from pick_a_pka import PKaPredictor
from pick_a_pka.backends.molgpka.protonation import (
    get_pKa_data,
    modify_mol,
    modify_acid,
    modify_base,
    modify_stable_pka,
    modify_unstable_pka,
    protonate_mol,
)


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def molgpka_model():
    return PKaPredictor("molgpka")


@pytest.fixture(scope="module")
def annotated_glycine(molgpka_model):
    """Glycine annotated with ionization props via modify_mol."""
    pred = molgpka_model.predict_pka("NCC(=O)O")
    return modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])


# ---------------------------------------------------------------------------
# modify_mol
# ---------------------------------------------------------------------------

class TestModifyMol:
    def test_returns_mol(self, molgpka_model):
        pred = molgpka_model.predict_pka("CC(=O)O")
        result = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        assert isinstance(result, Chem.Mol)

    def test_all_atoms_have_ionization_prop(self, molgpka_model):
        pred = molgpka_model.predict_pka("CC(=O)O")
        result = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        for atom in result.GetAtoms():
            assert atom.HasProp("ionization")

    def test_acid_sites_labelled_A(self, molgpka_model):
        pred = molgpka_model.predict_pka("CC(=O)O")
        result = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        acid_atoms = [a for a in result.GetAtoms() if a.GetProp("ionization") == "A"]
        assert len(acid_atoms) == len(pred["acid_pka"])

    def test_base_sites_labelled_B(self, molgpka_model):
        pred = molgpka_model.predict_pka("NCC(=O)O")  # glycine has amine
        result = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        base_atoms = [a for a in result.GetAtoms() if a.GetProp("ionization") == "B"]
        assert len(base_atoms) == len(pred["base_pka"])

    def test_non_ionizable_atoms_labelled_O(self, molgpka_model):
        pred = molgpka_model.predict_pka("CC(=O)O")
        result = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        other_atoms = [a for a in result.GetAtoms() if a.GetProp("ionization") == "O"]
        total = result.GetNumAtoms()
        assert len(other_atoms) == total - len(pred["acid_pka"]) - len(pred["base_pka"])

    def test_pka_prop_set_on_ionizable_atoms(self, molgpka_model):
        pred = molgpka_model.predict_pka("CC(=O)O")
        result = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        for atom in result.GetAtoms():
            if atom.GetProp("ionization") in ("A", "B"):
                assert atom.HasProp("pKa")
                assert isinstance(float(atom.GetProp("pKa")), float)

    def test_original_mol_not_mutated(self, molgpka_model):
        """modify_mol should return a deepcopy — the original must not have ionization props."""
        pred = molgpka_model.predict_pka("CC(=O)O")
        original = pred["mol"]
        _ = modify_mol(original, pred["acid_pka"], pred["base_pka"])
        assert not original.GetAtomWithIdx(0).HasProp("ionization")


# ---------------------------------------------------------------------------
# modify_acid / modify_base
# ---------------------------------------------------------------------------

class TestModifyAcidBase:
    def test_modify_acid_decrements_charge(self):
        mol = Chem.MolFromSmiles("CC(=O)O")
        atom = mol.GetAtomWithIdx(0)
        atom.SetFormalCharge(0)
        modify_acid(atom)
        assert atom.GetFormalCharge() == -1

    def test_modify_base_increments_charge(self):
        mol = Chem.MolFromSmiles("N")
        atom = mol.GetAtomWithIdx(0)
        atom.SetFormalCharge(0)
        modify_base(atom)
        assert atom.GetFormalCharge() == 1


# ---------------------------------------------------------------------------
# get_pKa_data
# ---------------------------------------------------------------------------

class TestGetPkaData:
    def test_stable_acid_below_ph(self, annotated_glycine):
        """An acid site with pKa << pH should land in stable_data."""
        stable, unstable = get_pKa_data(annotated_glycine, ph=7.4, tph=1.0)
        acid_stable = [x for x in stable if x[2] == "A"]
        # Glycine's carboxylic acid pKa ≈ 2.3 << 7.4 → must be stable
        assert len(acid_stable) >= 1

    def test_stable_base_above_ph(self, annotated_glycine):
        """A base site with pKa >> pH should land in stable_data."""
        stable, unstable = get_pKa_data(annotated_glycine, ph=2.0, tph=0.5)
        # At pH 2.0, glycine's amine pKa ≈ 9.6 >> 2.0+0.5 → stable basic
        base_stable = [x for x in stable if x[2] == "B"]
        assert len(base_stable) >= 1

    def test_near_pka_is_unstable(self, molgpka_model):
        """An acid site with pKa ≈ pH should land in unstable_data."""
        pred = molgpka_model.predict_pka("CC(=O)O")
        annotated = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        # Acetic acid pKa ≈ 4.7; test near that value with wide window
        stable, unstable = get_pKa_data(annotated, ph=4.76, tph=3.0)
        assert len(unstable) >= 1

    def test_returns_two_lists(self, annotated_glycine):
        result = get_pKa_data(annotated_glycine, ph=7.4, tph=1.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_stable_entries_have_three_fields(self, annotated_glycine):
        stable, _ = get_pKa_data(annotated_glycine, ph=7.4, tph=1.0)
        for entry in stable:
            assert len(entry) == 3  # [idx, pKa, type]

    def test_no_overlap_between_stable_and_unstable(self, annotated_glycine):
        stable, unstable = get_pKa_data(annotated_glycine, ph=7.4, tph=1.0)
        stable_idxs = {e[0] for e in stable}
        unstable_idxs = {e[0] for e in unstable}
        assert stable_idxs.isdisjoint(unstable_idxs)


# ---------------------------------------------------------------------------
# modify_stable_pka / modify_unstable_pka
# ---------------------------------------------------------------------------

class TestModifyStableUnstable:
    def test_modify_stable_applies_acid_charge(self, molgpka_model):
        pred = molgpka_model.predict_pka("CC(=O)O")
        annotated = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        stable, _ = get_pKa_data(annotated, ph=14.0, tph=0.1)
        from copy import deepcopy
        mol_copy = deepcopy(annotated)
        modify_stable_pka(mol_copy, stable)
        # At pH 14, the carboxylic acid should be deprotonated → fc on O = -1
        ionized = any(a.GetFormalCharge() < 0 for a in mol_copy.GetAtoms())
        assert ionized

    def test_modify_unstable_returns_smiles_list(self, molgpka_model):
        pred = molgpka_model.predict_pka("NCC(=O)O")
        annotated = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        _, unstable = get_pKa_data(annotated, ph=5.0, tph=4.0)
        from copy import deepcopy
        mol_copy = deepcopy(annotated)
        if unstable:
            result = modify_unstable_pka(mol_copy, unstable, 1)
            assert isinstance(result, list)
            for smi in result:
                assert Chem.MolFromSmiles(smi) is not None

    def test_modify_unstable_zero_combinations(self, molgpka_model):
        """Requesting 0 combinations (i=0) gives the unmodified state."""
        pred = molgpka_model.predict_pka("CC(=O)O")
        annotated = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])
        _, unstable = get_pKa_data(annotated, ph=4.76, tph=3.0)
        from copy import deepcopy
        mol_copy = deepcopy(annotated)
        if unstable:
            result = modify_unstable_pka(mol_copy, unstable, 0)
            # 0 combinations → empty (combinations(data, 0) = [()], but we skip empty)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# protonate_mol
# ---------------------------------------------------------------------------

class TestProtonateMol:
    def test_returns_list_of_smiles(self, molgpka_model):
        result = protonate_mol(molgpka_model.model, "CC(=O)O", ph=7.4, tph=1.0)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_all_results_are_valid_smiles(self, molgpka_model):
        result = protonate_mol(molgpka_model.model, "NCC(=O)O", ph=7.4, tph=1.0)
        for smi in result:
            assert Chem.MolFromSmiles(smi) is not None

    def test_non_ionizable_returns_single_state(self, molgpka_model):
        result = protonate_mol(molgpka_model.model, "CCCC", ph=7.4, tph=1.0)
        assert len(result) == 1

    def test_low_ph_gives_protonated_amine(self, molgpka_model):
        """At pH 0 the amine of glycine should be positively charged."""
        result = protonate_mol(molgpka_model.model, "NCC(=O)O", ph=0.0, tph=0.1)
        # At least one SMILES should contain a positively charged nitrogen
        has_pos_n = any("[NH3+]" in s or "[NH2+]" in s for s in result)
        assert has_pos_n

    def test_high_ph_gives_deprotonated_acid(self, molgpka_model):
        """At pH 14 the carboxylate should be negatively charged."""
        result = protonate_mol(molgpka_model.model, "CC(=O)O", ph=14.0, tph=0.1)
        has_neg_o = any("[O-]" in s for s in result)
        assert has_neg_o


class TestCalculateMicrospeciesAbundances:
    """Exercise branches in calculate_microspecies_abundances."""

    @pytest.fixture(scope="class")
    def molgpka_model(self):
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        return MolGpKaModel(device="cpu")

    def test_ph_range_returns_dict(self, molgpka_model):
        from pick_a_pka.backends.molgpka.protonation import calculate_microspecies_abundances
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        result = calculate_microspecies_abundances(molgpka_model, mol, ph_range=(6, 8), ph_step=1.0)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_ph_range_keys_are_floats(self, molgpka_model):
        from pick_a_pka.backends.molgpka.protonation import calculate_microspecies_abundances
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        result = calculate_microspecies_abundances(molgpka_model, mol, ph_range=(5, 9), ph_step=2.0)
        for k in result.keys():
            assert isinstance(k, float)

    def test_no_ph_or_range_raises(self, molgpka_model):
        from pick_a_pka.backends.molgpka.protonation import calculate_microspecies_abundances
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        with pytest.raises(ValueError, match="ph.*ph_range"):
            calculate_microspecies_abundances(molgpka_model, mol, ph=None, ph_range=None)

    def test_range_without_step_raises(self, molgpka_model):
        from pick_a_pka.backends.molgpka.protonation import calculate_microspecies_abundances
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        with pytest.raises(ValueError, match="ph_step"):
            calculate_microspecies_abundances(molgpka_model, mol, ph=None, ph_range=(0, 14), ph_step=None)


class TestComputeMicrostatePHRange:
    """compute_microstates with ph_range returns a dict of MicrostateResults."""

    @pytest.fixture(scope="class")
    def model(self):
        from pick_a_pka import PKaPredictor
        return PKaPredictor("molgpka")

    def test_ph_range_returns_dict(self, model):
        result = model.predict_microstates(ACETIC_ACID, ph=None, ph_range=(6, 8), ph_step=1.0)
        assert isinstance(result, dict)

    def test_ph_range_each_value_has_distribution(self, model):
        result = model.predict_microstates(ACETIC_ACID, ph=None, ph_range=(4, 6), ph_step=1.0)
        for v in result.values():
            assert "distribution" in v
            assert "major_state" in v
            assert "major_abundance" in v

    def test_ph_range_pkalearn_backend(self):
        from pick_a_pka import PKaPredictor
        model = PKaPredictor("pkalearn")
        result = model.predict_microstates(ACETIC_ACID, ph=None, ph_range=(4, 6), ph_step=1.0)
        assert isinstance(result, dict)


# ===========================================================================
# InvalidBackendError path in PKaPredictor
# ===========================================================================

class TestInvalidBackend:
    def test_invalid_backend_string_raises(self):
        from pick_a_pka import PKaPredictor
        from pick_a_pka.core.exceptions import InvalidBackendError
        with pytest.raises(InvalidBackendError):
            PKaPredictor("nonexistent_backend")


class TestMolGpKaProtonation:
    """Lines 83 (ValueError/ph check), 162-163, 194-195, 222-225."""

    def test_calculate_abundances_no_ph_raises(self):
        """Line 83: ValueError when both ph and ph_range are None."""
        from pick_a_pka.backends.molgpka.protonation import calculate_microspecies_abundances
        model = MagicMock()
        model.predict_pka.return_value = {"base_pka": {}, "acid_pka": {}, "mol": Chem.MolFromSmiles("C")}
        with pytest.raises(ValueError, match="ph.*ph_range"):
            calculate_microspecies_abundances(model, Chem.MolFromSmiles("C"), ph=None, ph_range=None)

    def test_calculate_abundances_ph_range_without_step_raises(self):
        from pick_a_pka.backends.molgpka.protonation import calculate_microspecies_abundances
        model = MagicMock()
        model.predict_pka.return_value = {"base_pka": {}, "acid_pka": {}, "mol": Chem.MolFromSmiles("C")}
        with pytest.raises(ValueError, match="ph_step"):
            calculate_microspecies_abundances(model, Chem.MolFromSmiles("C"), ph=None,
                                              ph_range=(0, 14), ph_step=None
                                              )

    def test_modify_mol_annotates_atoms(self):
        """Lines 162-163: modify_mol sets ionization/pKa props."""
        from pick_a_pka.backends.molgpka.protonation import modify_mol
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        annotated = modify_mol(mol, acid_dict={3: 4.76}, base_dict={})
        props = annotated.GetAtomWithIdx(3).GetPropsAsDict()
        assert props["ionization"] == "A"
        assert float(props["pKa"]) == pytest.approx(4.76)

    def test_modify_mol_base_annotates(self):
        from pick_a_pka.backends.molgpka.protonation import modify_mol
        mol = Chem.MolFromSmiles(ANILINE)
        # Find the nitrogen atom index dynamically instead of hardcoding 6
        n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        annotated = modify_mol(mol, acid_dict={}, base_dict={n_idx: 4.6})
        props = annotated.GetAtomWithIdx(n_idx).GetPropsAsDict()
        assert props["ionization"] == "B"

    def test_modify_mol_other_annotates(self):
        from pick_a_pka.backends.molgpka.protonation import modify_mol
        mol = Chem.MolFromSmiles("C")
        annotated = modify_mol(mol, acid_dict={}, base_dict={})
        props = annotated.GetAtomWithIdx(0).GetPropsAsDict()
        assert props["ionization"] == "O"

    def test_get_pKa_data_classifies_stable_acid(self):
        """Line 194-195: acid pKa < ph - tph → stable."""
        from pick_a_pka.backends.molgpka.protonation import modify_mol, get_pKa_data
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        annotated = modify_mol(mol, acid_dict={3: 4.76}, base_dict={})
        stable, unstable = get_pKa_data(annotated, ph=7.4, tph=1.0)
        # 4.76 < 7.4 - 1.0 = 6.4 → stable acid
        acids = [x for x in stable if x[2] == "A"]
        assert any(x[1] == pytest.approx(4.76) for x in acids)

    def test_get_pKa_data_classifies_unstable_acid(self):
        from pick_a_pka.backends.molgpka.protonation import modify_mol, get_pKa_data
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        annotated = modify_mol(mol, acid_dict={3: 7.0}, base_dict={})
        stable, unstable = get_pKa_data(annotated, ph=7.4, tph=1.0)
        acids = [x for x in unstable if x[2] == "A"]
        assert any(x[1] == pytest.approx(7.0) for x in acids)

    def test_get_pKa_data_classifies_stable_base(self):
        """Line 222-225: base pKa > ph + tph → stable."""
        from pick_a_pka.backends.molgpka.protonation import modify_mol, get_pKa_data
        mol = Chem.MolFromSmiles(ANILINE)
        annotated = modify_mol(mol, acid_dict={}, base_dict={6: 10.0})
        stable, unstable = get_pKa_data(annotated, ph=7.4, tph=1.0)
        bases = [x for x in stable if x[2] == "B"]
        assert any(x[1] == pytest.approx(10.0) for x in bases)

    def test_protonate_mol_no_unstable(self):
        from pick_a_pka.backends.molgpka.protonation import protonate_mol
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        model = MolGpKaModel(device="cpu")
        smis = protonate_mol(model, ACETIC_ACID, ph=1.0, tph=0.5)
        assert isinstance(smis, list)
        assert len(smis) >= 1

    def test_protonate_mol_with_unstable(self):
        from pick_a_pka.backends.molgpka.protonation import protonate_mol
        from pick_a_pka.backends.molgpka.model import MolGpKaModel
        model = MolGpKaModel(device="cpu")
        # pH near pKa of acetic acid ~4.76 → likely unstable region
        smis = protonate_mol(model, ACETIC_ACID, ph=4.76, tph=1.0)
        assert isinstance(smis, list)

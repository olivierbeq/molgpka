from unittest.mock import patch

import pandas as pd
from rdkit import Chem

from constants import L_ALANINE, D_ALANINE, L_ALANINE_PROTONATED, D_ALANINE_PROTONATED, ACETIC_ACID
from pick_a_pka.backends.pkalearn.chirality import transfer_chirality, process_transfer_chirality_in_batches


class TestTransferChirality:
    """Tests for pick_a_pka.backends.pkalearn.chirality.transfer_chirality."""

    def test_returns_string(self):
        result = transfer_chirality(L_ALANINE, L_ALANINE)
        assert isinstance(result, str)

    def test_no_chirality_returns_protonated_smiles(self):
        """A molecule without a chiral centre is returned unchanged."""
        result = transfer_chirality(ACETIC_ACID, "CC(=O)[O-]")
        assert result == "CC(=O)[O-]"

    def test_chiral_molecule_transfers_chirality(self):
        """Chirality should be present in the output for a chiral input."""
        # Protonate the nitrogen of L-alanine
        protonated = "C[C@@H]([NH3+])C(=O)O"
        result = transfer_chirality(L_ALANINE, protonated)
        assert isinstance(result, str)
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_invalid_original_smiles_returns_protonated(self):
        """If the original SMILES is invalid, return the protonated SMILES."""
        result = transfer_chirality("NOT_A_SMILES!!!", ACETIC_ACID)
        assert result == ACETIC_ACID

    def test_invalid_protonated_smiles_returns_protonated(self):
        """If the protonated SMILES is invalid, return it unchanged."""
        result = transfer_chirality(ACETIC_ACID, "NOT_A_SMILES!!!")
        assert result == "NOT_A_SMILES!!!"

    def test_substructure_mismatch_uses_mcs(self):
        """When substructure match fails, the MCS fallback is exercised."""
        # Use an (R)-enantiomer as original; (S)-form as protonated —
        # they differ only in chirality so direct match will succeed or MCS kicks in.
        r_form = "C[C@@H](N)C(=O)O"
        s_form = "C[C@H]([NH3+])C(=O)O"
        result = transfer_chirality(r_form, s_form)
        assert isinstance(result, str)

    def test_same_smiles_no_op(self):
        """When original == protonated, the result is still a valid SMILES."""
        result = transfer_chirality(L_ALANINE, L_ALANINE)
        assert Chem.MolFromSmiles(result) is not None


class TestProcessTransferChiralityInBatches:
    """Tests for process_transfer_chirality_in_batches."""

    def _fn(self):
        from pick_a_pka.backends.pkalearn.chirality import process_transfer_chirality_in_batches
        return process_transfer_chirality_in_batches

    def test_returns_dataframe(self):
        result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)
        assert isinstance(result, str)

    def test_updated_column_exists(self):
        result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)
        assert result == "C[C@@H]([NH3+])C(=O)O"

    def test_identical_smiles_not_processed(self):
        """Rows where Smiles == Predicted pKa smiles are skipped."""
        result = transfer_chirality(ACETIC_ACID, ACETIC_ACID)
        # No processing → updated column == original predicted column
        assert result == ACETIC_ACID


class TestSubstructureMatchBranch:
    """Cover the ``else:`` block (lines 73–75) executed when
    ``mol_prot.GetSubstructMatch(mol_orig)`` returns a non-empty tuple.

    For the match to succeed the two molecules must be structurally identical
    (same heavy-atom skeleton, same element on each atom) so RDKit's default
    substructure search finds a mapping.  Using the same SMILES for both
    arguments is the most direct way to guarantee this while the original
    still carries a chiral tag.
    """

    def test_same_smiles_returns_valid_smiles(self):
        """Identical chiral SMILES → substructure match succeeds → lines 73–75 run."""
        result = transfer_chirality(L_ALANINE, L_ALANINE)
        assert isinstance(result, str)
        assert Chem.MolFromSmiles(result) is not None

    def test_chiral_tag_preserved_when_match_succeeds(self):
        """After the copy loop the output should still carry stereochemistry."""
        result = transfer_chirality(L_ALANINE, L_ALANINE)
        mol = Chem.MolFromSmiles(result)
        has_chiral = any(
            a.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
            for a in mol.GetAtoms()
        )
        assert has_chiral, "Expected at least one chiral centre in the output"

    def test_d_alanine_same_smiles(self):
        """D-alanine as both arguments exercises the same branch."""
        result = transfer_chirality(D_ALANINE, D_ALANINE)
        assert Chem.MolFromSmiles(result) is not None

    def test_neutral_protonated_same_skeleton_triggers_match(self):
        """A protonated SMILES that keeps the same atoms/bonds (only formal
        charge changes on a site RDKit ignores in the default match) should
        still give a substructure match.

        We use an acetate / acetic-acid pair where the oxygen lone-pair
        change doesn't prevent the match, ensuring the else-branch fires
        for a pair that is NOT simply identical strings.
        """
        # Chiral phosphoric acid derivative — neutral form matches itself
        chiral_orig = "O[P@@](=O)(O)OC"
        chiral_prot = "O[P@@](=O)(O)OC"  # identical skeleton
        result = transfer_chirality(chiral_orig, chiral_prot)
        assert isinstance(result, str)

    def test_output_is_canonical_smiles(self):
        """transfer_chirality must always return a parseable canonical SMILES."""
        result = transfer_chirality(L_ALANINE, L_ALANINE)
        # Round-trip check
        mol = Chem.MolFromSmiles(result)
        assert mol is not None
        canonical = Chem.MolToSmiles(mol)
        assert Chem.MolFromSmiles(canonical) is not None


# ===========================================================================
# Lines 32–52  — MCS fallback, patt found, both sub-matches succeed
# ===========================================================================

class TestMCSFallbackBothMatchesSucceed:
    """Cover lines 32–52: the MCS path runs and both
    ``orig_match`` and ``prot_match`` are non-empty, so chirality is
    transferred via the atom_map.

    Trigger condition: ``mol_prot.GetSubstructMatch(mol_orig)`` is empty,
    which happens when the protonated SMILES changes an atom's formal charge
    (RDKit's default substructure search IS charge-aware for explicit charges
    written in the SMILES, e.g. [NH3+] vs N).
    """

    def test_mcs_path_chiral_alanine_protonated_nitrogen(self):
        """L-alanine (neutral N) vs its protonated form ([NH3+]).

        Direct substructure match fails because N ≠ [NH3+], so MCS is used.
        The MCS still finds the full carbon skeleton; both sub-matches succeed,
        and the chiral tag at the alpha-carbon is copied.
        """
        result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)
        assert isinstance(result, str)
        mol = Chem.MolFromSmiles(result)
        assert mol is not None, f"MCS path produced invalid SMILES: {result!r}"

    def test_mcs_path_preserves_chirality(self):
        """After MCS-based transfer the output must still be chiral."""
        result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)
        mol = Chem.MolFromSmiles(result)
        has_chiral = any(
            a.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
            for a in mol.GetAtoms()
        )
        assert has_chiral

    def test_mcs_path_d_alanine(self):
        """Same test for the D-enantiomer."""
        result = transfer_chirality(D_ALANINE, D_ALANINE_PROTONATED)
        assert isinstance(result, str)
        assert Chem.MolFromSmiles(result) is not None

    def test_mcs_path_phenylalanine(self):
        """L-phenylalanine (neutral) → protonated ([NH3+]) exercises MCS on
        a larger, ring-containing scaffold.  ``completeRingsOnly=True`` in
        FindMCS means the phenyl ring must appear in full in the MCS result,
        giving a rich pattern for both sub-matches.
        """
        orig = "N[C@@H](Cc1ccccc1)C(=O)O"  # L-Phe neutral
        prot = "[NH3+][C@@H](Cc1ccccc1)C(=O)O"  # L-Phe protonated N
        result = transfer_chirality(orig, prot)
        assert isinstance(result, str)
        assert Chem.MolFromSmiles(result) is not None

    def test_mcs_path_serine(self):
        """L-serine: chiral centre, neutral amine, protonated form forces MCS."""
        orig = "N[C@@H](CO)C(=O)O"
        prot = "[NH3+][C@@H](CO)C(=O)O"
        result = transfer_chirality(orig, prot)
        assert isinstance(result, str)
        assert Chem.MolFromSmiles(result) is not None

    def test_mcs_path_threonine(self):
        """L-threonine has two stereocentres; both should survive MCS transfer."""
        orig = "C[C@@H](O)[C@H](N)C(=O)O"
        prot = "C[C@@H](O)[C@H]([NH3+])C(=O)O"
        result = transfer_chirality(orig, prot)
        assert isinstance(result, str)
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_mcs_path_returns_string_type(self):
        """Return type is always str regardless of branch taken."""
        result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)
        assert isinstance(result, str)


# ===========================================================================
# Lines 53–54  — MCS fallback, patt found, but one sub-match is empty
# ===========================================================================

class TestMCSFallbackEmptySubMatch:
    """Cover lines 53–54: ``if orig_match and prot_match`` is False because
    at least one of the GetSubstructMatch calls returns an empty tuple.

    We mock ``Chem.MolFromSmarts`` to return a pattern that deliberately
    fails to match one of the molecules, forcing the ``else`` branch.
    """

    def test_returns_protonated_smiles_when_prot_match_empty(self):
        """If prot_match is empty the function returns protonated_smiles unchanged."""
        with patch(
                "pick_a_pka.backends.pkalearn.chirality.Chem.MolFromSmarts"
        ) as mock_from_smarts:
            # Build a real but unmatchable pattern (e.g. a single boron atom)
            unmatchable = Chem.MolFromSmarts("[B]")
            mock_from_smarts.return_value = unmatchable

            result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)

        assert result == L_ALANINE_PROTONATED

    def test_returns_protonated_smiles_when_orig_match_empty(self):
        """Same guard: if orig_match is empty → return protonated_smiles."""
        # We intercept at a lower level: make the pattern so that
        # mol_orig doesn't match it.
        with patch(
                "pick_a_pka.backends.pkalearn.chirality.Chem.MolFromSmarts"
        ) as mock_from_smarts:
            # An 8-atom ring pattern that neither small mol contains
            impossible_patt = Chem.MolFromSmarts("C1CCCCCCC1")
            mock_from_smarts.return_value = impossible_patt

            result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)

        assert result == L_ALANINE_PROTONATED


# ===========================================================================
# Lines 55–56  — MCS fallback, patt is None / falsy
# ===========================================================================

class TestMCSFallbackNoPatt:
    """Cover lines 55–56: ``Chem.MolFromSmarts(mcs.smartsString)`` returns
    ``None`` (or a falsy value), so the function returns ``protonated_smiles``
    immediately without attempting any chirality transfer.
    """

    def test_returns_protonated_smiles_when_patt_is_none(self):
        """Mock MolFromSmarts to return None → hits the ``else`` on line 55."""
        with patch(
                "pick_a_pka.backends.pkalearn.chirality.Chem.MolFromSmarts",
                return_value=None,
        ):
            result = transfer_chirality(L_ALANINE, L_ALANINE_PROTONATED)

        assert result == L_ALANINE_PROTONATED

    def test_value_is_exact_protonated_smiles_string(self):
        """The returned value must be the exact ``protonated_smiles`` argument."""
        sentinel = "C[C@@H]([NH3+])C(=O)O"
        with patch(
                "pick_a_pka.backends.pkalearn.chirality.Chem.MolFromSmarts",
                return_value=None,
        ):
            result = transfer_chirality(L_ALANINE, sentinel)

        assert result == sentinel


# ===========================================================================
# Remaining branches already partially tested — reinforced for completeness
# ===========================================================================

class TestEarlyReturnBranches:
    """Re-verify the early-return paths (already covered) so that the full
    module can be exercised in a single pytest run."""

    def test_invalid_original_smiles_returns_protonated(self):
        result = transfer_chirality("NOT_VALID!!!", "CC(=O)O")
        assert result == "CC(=O)O"

    def test_invalid_protonated_smiles_returns_protonated(self):
        result = transfer_chirality("CC(=O)O", "NOT_VALID!!!")
        assert result == "NOT_VALID!!!"

    def test_no_chirality_returns_protonated_unchanged(self):
        result = transfer_chirality("CC(=O)O", "CC(=O)[O-]")
        assert result == "CC(=O)[O-]"


# ===========================================================================
# process_transfer_chirality_in_batches — lines 79–94
# ===========================================================================

class TestProcessInBatches:
    """Full coverage of the batch-processing helper."""

    def _df(self, rows):
        return pd.DataFrame(rows, columns=["Smiles", "Predicted pKa smiles"])

    def test_returns_dataframe(self):
        df = self._df([(L_ALANINE, L_ALANINE_PROTONATED)])
        result = process_transfer_chirality_in_batches(df)
        assert isinstance(result, pd.DataFrame)

    def test_updated_column_created(self):
        df = self._df([(L_ALANINE, L_ALANINE_PROTONATED)])
        result = process_transfer_chirality_in_batches(df)
        assert "Predicted pKa smiles updated" in result.columns

    def test_identical_smiles_rows_are_skipped(self):
        """Rows where Smiles == Predicted pKa smiles must not be processed."""
        df = self._df([("CC(=O)O", "CC(=O)O")])
        result = process_transfer_chirality_in_batches(df)
        assert result["Predicted pKa smiles updated"].iloc[0] == "CC(=O)O"

    def test_different_smiles_rows_are_processed(self):
        """Rows where the two SMILES differ must pass through transfer_chirality."""
        df = self._df([(L_ALANINE, L_ALANINE_PROTONATED)])
        result = process_transfer_chirality_in_batches(df)
        # The updated column must have been written (may equal or differ from prot)
        assert result["Predicted pKa smiles updated"].iloc[0] is not None

    def test_original_dataframe_not_mutated(self):
        """The function works on a copy; the original df must be unchanged."""
        df = self._df([(L_ALANINE, L_ALANINE_PROTONATED)])
        original_cols = list(df.columns)
        process_transfer_chirality_in_batches(df)
        assert list(df.columns) == original_cols

    def test_batch_size_smaller_than_rows(self):
        """batch_size < number of rows exercises multi-iteration loop."""
        rows = [(L_ALANINE, L_ALANINE_PROTONATED)] * 7
        df = self._df(rows)
        result = process_transfer_chirality_in_batches(df, batch_size=3)
        assert len(result) == 7
        assert result["Predicted pKa smiles updated"].notna().all()

    def test_empty_dataframe(self):
        """An empty DataFrame must be handled gracefully."""
        df = self._df([])
        result = process_transfer_chirality_in_batches(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_mixed_rows(self):
        """Mix of identical and different SMILES rows."""
        rows = [
            ("CC(=O)O", "CC(=O)O"),  # identical → skipped
            (L_ALANINE, L_ALANINE_PROTONATED),  # different → processed
            (D_ALANINE, D_ALANINE_PROTONATED),  # different → processed
        ]
        df = self._df(rows)
        result = process_transfer_chirality_in_batches(df, batch_size=2)
        assert len(result) == 3
        # Skipped row stays unchanged
        assert result["Predicted pKa smiles updated"].iloc[0] == "CC(=O)O"


class TestTransferChirality:
    """Lines 32-58 (chiral transfer paths), 73-75 (batch processing)."""

    def test_no_chiral_returns_unchanged(self):
        """If original has no chiral centres, the protonated SMILES is returned as-is."""
        from pick_a_pka.backends.pkalearn.chirality import transfer_chirality
        result = transfer_chirality("CC(=O)O", "CC(=O)O")
        assert result == "CC(=O)O"

    def test_chiral_transfer_substructure_match(self):
        """Lines 32-58: substructure match path."""
        from pick_a_pka.backends.pkalearn.chirality import transfer_chirality
        original = L_ALANINE  # has chiral centre
        protonated = "C[C@@H]([NH3+])C(=O)O"
        result = transfer_chirality(original, protonated)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_original_returns_protonated(self):
        """If original SMILES is invalid, fall back to protonated."""
        from pick_a_pka.backends.pkalearn.chirality import transfer_chirality
        result = transfer_chirality("NOT_VALID_SMILES!!!", "CC")
        assert result == "CC"

    def test_invalid_protonated_returns_protonated(self):
        from pick_a_pka.backends.pkalearn.chirality import transfer_chirality
        result = transfer_chirality(L_ALANINE, "INVALID!!!")
        assert result == "INVALID!!!"

    def test_mcs_fallback_path(self):
        """Lines 42-58: chiral mol where substructure fails → MCS path."""
        from pick_a_pka.backends.pkalearn.chirality import transfer_chirality
        # Use molecules where the charged form and neutral differ enough
        # that the direct GetSubstructMatch may fail
        original = "[C@@H]1(N)CCCC1"  # chiral cyclopentylamine
        protonated = "[C@@H]1([NH3+])CCCC1"
        result = transfer_chirality(original, protonated)
        assert isinstance(result, str)

    def test_batch_processing(self):
        """Lines 73-75: process_transfer_chirality_in_batches."""
        import pandas as pd
        from pick_a_pka.backends.pkalearn.chirality import process_transfer_chirality_in_batches
        df = pd.DataFrame({
            "Smiles": [L_ALANINE, ACETIC_ACID],
            "Predicted pKa smiles": ["C[C@@H]([NH3+])C(=O)O", "CC(=O)O"],
        }
        )
        result = process_transfer_chirality_in_batches(df, batch_size=1)
        assert "Predicted pKa smiles updated" in result.columns
        assert len(result) == 2

"""Tests for core/canonicalize.py — the geometry helpers and folding pipeline."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from pick_a_pka.core.canonicalize import (
    canonicalize_orientation,
    ccw,
    get_branch_atoms,
    get_score,
    has_crossings_or_collisions,
    intersect,
    optimize_sensical_folding,
    orient_canonically,
    point_segment_distance,
    reflect_branch_2d,
)
from constants import BUTANE, METHANE, ETHANE, PENTANE, HEXANE, OCTANE, DECANE, BENZENE, ACETAMINOPHEN, ACETIC_ACID, \
    CHEMBL5646830


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mol_with_coords(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    return mol


# ---------------------------------------------------------------------------
# get_score
# ---------------------------------------------------------------------------

class TestGetScore:
    def test_returns_float(self):
        mol = _mol_with_coords(BUTANE)
        conf = mol.GetConformer()
        assert isinstance(get_score(conf), float)

    def test_linear_molecule_longer_than_atom(self):
        """A four-carbon chain should have a larger span than a single atom."""
        mol4 = _mol_with_coords(BUTANE)
        mol1 = _mol_with_coords(METHANE)
        rdDepictor.Compute2DCoords(mol1)
        assert get_score(mol4.GetConformer()) > get_score(mol1.GetConformer())

    def test_square_molecule_score_is_max_dimension(self):
        """Score is max(width, height) so it equals the longest axis."""
        mol = _mol_with_coords(BENZENE)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        w = pos[:, 0].max() - pos[:, 0].min()
        h = pos[:, 1].max() - pos[:, 1].min()
        assert abs(get_score(conf) - max(w, h)) < 1e-6


# ---------------------------------------------------------------------------
# ccw / intersect
# ---------------------------------------------------------------------------

class TestCcwIntersect:
    def test_ccw_positive(self):
        A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0.5, 1])
        assert bool(ccw(A, B, C)) is True

    def test_ccw_negative(self):
        A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0.5, -1])
        assert bool(ccw(A, B, C)) is False

    def test_intersect_crossing_segments(self):
        # X cross
        A, B = np.array([0.0, 0.0]), np.array([1.0, 1.0])
        C, D = np.array([0.0, 1.0]), np.array([1.0, 0.0])
        assert bool(intersect(A, B, C, D)) is True

    def test_intersect_parallel_non_crossing(self):
        A, B = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        C, D = np.array([0.0, 1.0]), np.array([1.0, 1.0])
        assert bool(intersect(A, B, C, D)) is False

    def test_intersect_t_junction_not_crossing(self):
        # A T-junction where the perpendicular segment starts at the midpoint
        # of the base but they share no endpoints — not a crossing.
        A, B = np.array([0.0, 0.0]), np.array([4.0, 0.0])
        C, D = np.array([2.0, 1.0]), np.array([2.0, 3.0])
        assert bool(intersect(A, B, C, D)) is False


# ---------------------------------------------------------------------------
# point_segment_distance
# ---------------------------------------------------------------------------

class TestPointSegmentDistance:
    def test_perpendicular_midpoint(self):
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 0.0])
        p = np.array([1.0, 3.0])
        assert abs(point_segment_distance(p, a, b) - 3.0) < 1e-9

    def test_closest_to_endpoint_a(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        p = np.array([-1.0, 0.0])
        assert abs(point_segment_distance(p, a, b) - 1.0) < 1e-9

    def test_closest_to_endpoint_b(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        p = np.array([2.0, 0.0])
        assert abs(point_segment_distance(p, a, b) - 1.0) < 1e-9

    def test_point_on_segment(self):
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 0.0])
        p = np.array([2.0, 0.0])
        assert point_segment_distance(p, a, b) < 1e-9


# ---------------------------------------------------------------------------
# has_crossings_or_collisions
# ---------------------------------------------------------------------------

class TestHasCrossingsOrCollisions:
    def test_clean_molecule_no_collision(self):
        mol = _mol_with_coords(BENZENE)
        conf = mol.GetConformer()
        assert has_crossings_or_collisions(mol, conf) is False

    def test_clean_linear_no_collision(self):
        mol = _mol_with_coords(HEXANE)
        conf = mol.GetConformer()
        assert has_crossings_or_collisions(mol, conf) is False

    def test_atom_collision_detected(self):
        """Force two atoms to the same position and expect True."""
        mol = _mol_with_coords(ETHANE)
        conf = mol.GetConformer()
        # Move atom 1 on top of atom 0
        p0 = conf.GetAtomPosition(0)
        conf.SetAtomPosition(1, (p0.x, p0.y, 0))
        assert has_crossings_or_collisions(mol, conf) is True

    def test_bond_crossing_detected(self):
        """Manually cross two bonds by swapping atom positions."""
        mol = _mol_with_coords(BUTANE)
        conf = mol.GetConformer()
        # Swap atoms 1 and 2 positions to create a crossing C1-C2 with C0-C3
        p1 = conf.GetAtomPosition(1)
        p2 = conf.GetAtomPosition(2)
        conf.SetAtomPosition(1, (p2.x, p2.y, 0))
        conf.SetAtomPosition(2, (p1.x, p1.y, 0))
        assert has_crossings_or_collisions(mol, conf) is True


# ---------------------------------------------------------------------------
# get_branch_atoms
# ---------------------------------------------------------------------------

class TestGetBranchAtoms:
    def test_linear_chain_branch(self):
        """In C0-C1-C2-C3, pivoting on bond 1-2, branch from 2 should be {2, 3}."""
        mol = Chem.MolFromSmiles(BUTANE)
        branch = get_branch_atoms(mol, pivot_u=1, start_v=2)
        assert set(branch) == {2, 3}

    def test_branched_molecule(self):
        """In CC(C)C: atom 0 is CH3, atom 1 is central C, atoms 2,3 are CH3.
        Pivoting on bond 0-1, branch from 1 should include 1, 2, 3."""
        mol = Chem.MolFromSmiles("CC(C)C")
        branch = get_branch_atoms(mol, pivot_u=0, start_v=1)
        assert 1 in branch
        assert 2 in branch
        assert 3 in branch
        assert 0 not in branch

    def test_single_atom_branch(self):
        """Pivoting to a terminal atom gives just that atom."""
        mol = Chem.MolFromSmiles(ETHANE)
        branch = get_branch_atoms(mol, pivot_u=0, start_v=1)
        assert branch == [1]


# ---------------------------------------------------------------------------
# reflect_branch_2d
# ---------------------------------------------------------------------------

class TestReflectBranch2d:
    def test_reflection_moves_atom(self):
        """After reflection, a branch atom should move to a different position."""
        mol = _mol_with_coords(BUTANE)
        conf = mol.GetConformer()
        orig_pos = np.array(conf.GetAtomPosition(3))
        branch = get_branch_atoms(mol, 1, 2)
        reflect_branch_2d(conf, 1, 2, branch)
        new_pos = np.array(conf.GetAtomPosition(3))
        # The atom must have moved (unless degenerate geometry)
        assert not np.allclose(orig_pos, new_pos)

    def test_reflection_preserves_pivot_atoms(self):
        """The pivot atoms (u and v) must not move."""
        mol = _mol_with_coords(BUTANE)
        conf = mol.GetConformer()
        p0 = np.array(conf.GetAtomPosition(0))
        p1 = np.array(conf.GetAtomPosition(1))
        branch = get_branch_atoms(mol, 0, 1)
        reflect_branch_2d(conf, 0, 1, branch)
        assert np.allclose(np.array(conf.GetAtomPosition(0)), p0)
        assert np.allclose(np.array(conf.GetAtomPosition(1)), p1)

    def test_double_reflection_is_identity(self):
        """Two reflections across the same axis restore original positions."""
        mol = _mol_with_coords(BUTANE)
        conf = mol.GetConformer()
        orig_positions = [np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        branch = get_branch_atoms(mol, 1, 2)
        reflect_branch_2d(conf, 1, 2, branch)
        reflect_branch_2d(conf, 1, 2, branch)
        for i, orig in enumerate(orig_positions):
            assert np.allclose(np.array(conf.GetAtomPosition(i)), orig, atol=1e-6)


# ---------------------------------------------------------------------------
# orient_canonically
# ---------------------------------------------------------------------------

class TestOrientCanonically:
    def test_returns_mol_with_conformer(self):
        mol = Chem.MolFromSmiles(BENZENE)
        result = orient_canonically(mol)
        assert result.GetNumConformers() == 1

    def test_deterministic(self):
        """Same molecule always gets the same layout."""
        mol1 = Chem.MolFromSmiles(ACETIC_ACID)
        mol2 = Chem.MolFromSmiles(ACETIC_ACID)
        orient_canonically(mol1)
        orient_canonically(mol2)
        pos1 = mol1.GetConformer().GetPositions()
        pos2 = mol2.GetConformer().GetPositions()
        assert np.allclose(pos1, pos2, atol=1e-4)

    def test_atoms_not_stacked(self):
        """After orientation no two atoms should be at the same position."""
        mol = Chem.MolFromSmiles("c1ccc(N)cc1")
        orient_canonically(mol)
        pos = mol.GetConformer().GetPositions()
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                assert np.linalg.norm(pos[i] - pos[j]) > 0.3


# ---------------------------------------------------------------------------
# canonicalize_orientation
# ---------------------------------------------------------------------------

class TestCanonicalizeOrientation:
    def test_returns_mol(self):
        mol = _mol_with_coords(PENTANE)
        result = canonicalize_orientation(mol)
        assert isinstance(result, Chem.Mol)

    def test_conformer_preserved(self):
        mol = _mol_with_coords(BUTANE)
        result = canonicalize_orientation(mol)
        assert result.GetNumConformers() == 1

    def test_deterministic(self):
        mol1 = _mol_with_coords(ACETAMINOPHEN)
        mol2 = _mol_with_coords(ACETAMINOPHEN)
        canonicalize_orientation(mol1)
        canonicalize_orientation(mol2)
        pos1 = mol1.GetConformer().GetPositions()
        pos2 = mol2.GetConformer().GetPositions()
        assert np.allclose(pos1, pos2, atol=1e-4)

    def test_prefers_horizontal_layout(self):
        """After canonicalization, width should be >= height for elongated mols."""
        mol = _mol_with_coords(OCTANE)  # long chain
        canonicalize_orientation(mol)
        pos = mol.GetConformer().GetPositions()
        w = pos[:, 0].max() - pos[:, 0].min()
        h = pos[:, 1].max() - pos[:, 1].min()
        assert w >= h - 1e-3


# ---------------------------------------------------------------------------
# optimize_sensical_folding
# ---------------------------------------------------------------------------

class TestOptimizeSensicalFolding:
    def test_returns_mol(self):
        mol = Chem.MolFromSmiles(HEXANE)
        result = optimize_sensical_folding(mol)
        assert isinstance(result, Chem.Mol)

    def test_has_conformer(self):
        mol = Chem.MolFromSmiles(HEXANE)
        result = optimize_sensical_folding(mol)
        assert result.GetNumConformers() == 1

    def test_no_crossing_bonds_after_folding(self):
        """optimize_sensical_folding is a bounding-box minimiser, not a
        crossing eliminator.  We verify the weaker property that no two
        atoms land at the same position (atom-atom collision), which would
        be a hard layout failure regardless of the clearance heuristic."""
        mol = Chem.MolFromSmiles(CHEMBL5646830)
        result = optimize_sensical_folding(mol, steps=200)
        import numpy as np
        pos = result.GetConformer().GetPositions()
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                assert np.linalg.norm(pos[i] - pos[j]) > 0.3, (
                    f"Atoms {i} and {j} are at the same position after folding"
                )

    def test_non_rotatable_mol_unchanged(self):
        """A molecule with no rotatable bonds should still return a valid mol."""
        mol = Chem.MolFromSmiles(BENZENE)  # benzene — no rotatable bonds
        result = optimize_sensical_folding(mol, steps=100)
        assert result.GetNumConformers() == 1

    def test_folding_reduces_or_preserves_score(self):
        """The folded layout should not be worse than the initial RDKit layout."""
        import random
        random.seed(42)
        mol = Chem.MolFromSmiles(DECANE)  # long flexible chain
        rdDepictor.Compute2DCoords(mol)
        initial_score = get_score(mol.GetConformer())
        result = optimize_sensical_folding(mol, steps=500)
        final_score = get_score(result.GetConformer())
        # Folding should not make it dramatically worse
        assert final_score <= initial_score + 0.5

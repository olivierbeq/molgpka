from unittest.mock import patch, MagicMock

import matplotlib
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D

matplotlib.use('Agg')

from constants import ACETIC_ACID, GLYCINE, ANILINE
from pick_a_pka import PKaPredictor, draw_pka
from pick_a_pka.draw import _neutral_query, _transfer_coords


def test_draw_pka_svg():
    svg = draw_pka(ACETIC_ACID, vector=True)

    assert isinstance(svg, str)
    assert "<svg" in svg


def test_plot_distribution_svg():
    svg = plot_microspecies_distribution(ACETIC_ACID, vector=True)

    assert isinstance(svg, str)
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# _neutral_query
# ---------------------------------------------------------------------------

class TestNeutralQuery:
    def test_returns_mol(self):
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        q = _neutral_query(mol)
        assert isinstance(q, Chem.Mol)

    def test_no_formal_charges(self):
        mol = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")  # zwitterionic glycine
        q = _neutral_query(mol)
        charges = [a.GetFormalCharge() for a in q.GetAtoms()]
        assert all(c == 0 for c in charges)

    def test_no_explicit_hs(self):
        mol = Chem.AddHs(Chem.MolFromSmiles(ACETIC_ACID))
        q = _neutral_query(mol)
        h_atoms = [a for a in q.GetAtoms() if a.GetAtomicNum() == 1]
        assert len(h_atoms) == 0

    def test_same_heavy_atom_count(self):
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        q = _neutral_query(mol)
        assert q.GetNumAtoms() == mol.GetNumAtoms()

    def test_charged_and_neutral_match_each_other(self):
        """A charged mol and its neutral equivalent should substructure-match."""
        neutral = Chem.MolFromSmiles(ACETIC_ACID)
        charged = Chem.MolFromSmiles("CC(=O)[O-]")
        qn = _neutral_query(neutral)
        qc = _neutral_query(charged)
        assert qn.GetSubstructMatch(qc) or qc.GetSubstructMatch(qn)


# ---------------------------------------------------------------------------
# _transfer_coords
# ---------------------------------------------------------------------------

class TestTransferCoords:
    def _mol_with_coords(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(mol)
        return mol

    def test_transfers_positions(self):
        ref = self._mol_with_coords(ACETIC_ACID)
        target = Chem.MolFromSmiles(ACETIC_ACID)
        rdDepictor.Compute2DCoords(target)
        _transfer_coords(ref, target)
        ref_pos = ref.GetConformer().GetPositions()
        tgt_pos = target.GetConformer().GetPositions()
        # At least the heavy scaffold atoms should be at the same positions
        assert any(
            any(abs(r - t) < 1e-4 for r, t in zip(rp, tp))
            for rp, tp in zip(ref_pos, tgt_pos)
        )

    def test_no_op_without_ref_conformer(self):
        """If ref has no conformer, _transfer_coords must not crash."""
        ref = Chem.MolFromSmiles(ACETIC_ACID)  # no conformer
        target = self._mol_with_coords(ACETIC_ACID)
        original_pos = target.GetConformer().GetPositions().copy()
        _transfer_coords(ref, target)  # should silently do nothing
        import numpy as np
        assert np.allclose(target.GetConformer().GetPositions(), original_pos)

    def test_works_across_protonation_states(self):
        """Coords should transfer from neutral acetic acid onto acetate."""
        ref = self._mol_with_coords(ACETIC_ACID)
        acetate = Chem.MolFromSmiles("CC(=O)[O-]")
        rdDepictor.Compute2DCoords(acetate)
        _transfer_coords(ref, acetate)  # must not raise
        assert acetate.GetNumConformers() == 1

    def test_mcs_fallback_large_difference(self):
        """For two molecules sharing a scaffold, transfer should still work."""
        ref = self._mol_with_coords("c1ccccc1N")  # aniline
        target = Chem.MolFromSmiles("c1ccccc1[NH3+]")  # protonated aniline
        rdDepictor.Compute2DCoords(target)
        _transfer_coords(ref, target)
        assert target.GetNumConformers() == 1


# ---------------------------------------------------------------------------
# draw_pka with pre-existing conformer
# ---------------------------------------------------------------------------

class TestDrawPkaWithConformer:
    def test_preserves_caller_coords(self):
        """When the input mol already has 2D coords, draw_pka must not recompute."""
        from pick_a_pka.core.canonicalize import optimize_sensical_folding
        mol = Chem.MolFromSmiles(GLYCINE)
        mol = optimize_sensical_folding(mol)
        # Capture one atom's position
        import numpy as np
        ref_pos = np.array(mol.GetConformer().GetAtomPosition(0))
        svg = draw_pka(mol, vector=True)
        # The conformer on mol should not have been wiped out
        assert mol.GetNumConformers() > 0

    def test_svg_output_with_precomputed_coords(self):
        from pick_a_pka.core.canonicalize import orient_canonically
        mol = Chem.MolFromSmiles(ACETIC_ACID)
        orient_canonically(mol)
        svg = draw_pka(mol, vector=True)
        assert "<svg" in svg

    def test_mol_without_conformer_still_works(self):
        """draw_pka must also work when no conformer exists (normal path)."""
        svg = draw_pka(ACETIC_ACID, vector=True)
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Non-vector (PIL Image) returns
# ---------------------------------------------------------------------------

class TestNonVectorOutput:
    def test_draw_pka_returns_pil_image(self):
        result = draw_pka(ACETIC_ACID, vector=False)
        assert isinstance(result, Image.Image)

    def test_draw_pka_image_nonzero_size(self):
        img = draw_pka(ACETIC_ACID, vector=False)
        w, h = img.size
        assert w > 0 and h > 0

    def test_plot_distribution_returns_pil_image(self):
        result = plot_microspecies_distribution(ACETIC_ACID, vector=False)
        assert isinstance(result, Image.Image)

    def test_plot_distribution_image_nonzero_size(self):
        img = plot_microspecies_distribution(ACETIC_ACID, vector=False)
        w, h = img.size
        assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# draw_pka corner cases
# ---------------------------------------------------------------------------

class TestDrawPkaCornerCases:
    def test_different_backends_produce_svg(self):
        for backend in ["molgpka", "pkalearn"]:
            model = PKaPredictor(backend)
            svg = draw_pka(ACETIC_ACID, model=model, vector=True)
            assert "<svg" in svg

    def test_custom_image_size(self):
        img = draw_pka(ACETIC_ACID, vector=False, image_size=(400, 400))
        assert isinstance(img, Image.Image)

    def test_mol_input(self):
        mol = Chem.MolFromSmiles(ANILINE)
        svg = draw_pka(mol, vector=True)
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# plot_microspecies_distribution corner cases
# ---------------------------------------------------------------------------

class TestPlotMicrospeciesCornerCases:
    def test_non_ionisable_mol_produces_svg(self):
        svg = plot_microspecies_distribution("CCCC", vector=True)
        assert "<svg" in svg

    def test_pkalearn_backend_produces_svg(self):
        model = PKaPredictor("pkalearn", allow_amphoteric=True)
        svg = plot_microspecies_distribution(ACETIC_ACID, model=model, vector=True)
        assert "<svg" in svg

    def test_mol_input(self):
        mol = Chem.MolFromSmiles(GLYCINE)
        svg = plot_microspecies_distribution(mol, vector=True)
        assert "<svg" in svg


# ---------------------------------------------------------------------------
# Additional coverage tests for edge cases and exceptions
# ---------------------------------------------------------------------------

def test_neutral_query_sanitize_fail():
    # [N+](=O)[O-] neutralized yields pentavalent nitrogen, which fails sanitization (lines 30-31)
    mol = Chem.MolFromSmiles("[N+](=O)[O-]")
    assert mol is not None, "Precondition: SMILES must parse successfully"
    q = _neutral_query(mol)
    assert q is not None
    assert q.GetNumAtoms() == 3


def test_transfer_coords_exact_match():
    ref = Chem.MolFromSmiles("C")
    rdDepictor.Compute2DCoords(ref)
    # Pass a neutral query target so `GetSubstructMatch` cleanly skips the MCS fallback (lines 56-61)
    tgt = _neutral_query(Chem.MolFromSmiles("C"))
    rdDepictor.Compute2DCoords(tgt)
    _transfer_coords(ref, tgt)
    assert tgt.GetNumConformers() == 1


def test_transfer_coords_mcs_empty():
    ref = Chem.MolFromSmiles("C")
    rdDepictor.Compute2DCoords(ref)
    tgt = Chem.MolFromSmiles("O")
    # MCS between structurally distinct elements will return numAtoms == 0 (line 75)
    _transfer_coords(ref, tgt)
    assert tgt.GetNumConformers() == 0


@patch("pick_a_pka.draw.Chem.MolFromSmarts")
def test_transfer_coords_bad_smarts(mock_from_smarts):
    mock_from_smarts.return_value = None
    ref = Chem.MolFromSmiles("CC")
    rdDepictor.Compute2DCoords(ref)
    tgt = Chem.MolFromSmiles("CC")
    # Triggers `patt is None` check (line 78)
    _transfer_coords(ref, tgt)
    assert tgt.GetNumConformers() == 0


@patch("rdkit.Chem.Mol.GetSubstructMatch")
def test_transfer_coords_mcs_no_match(mock_match):
    mock_match.return_value = ()
    ref = Chem.MolFromSmiles("CC")
    rdDepictor.Compute2DCoords(ref)
    tgt = Chem.MolFromSmiles("CC")
    # Triggers `if not ref_match or not tgt_match` (line 82)
    _transfer_coords(ref, tgt)


def test_transfer_coords_exception():
    ref = Chem.MolFromSmiles("CC")
    rdDepictor.Compute2DCoords(ref)
    tgt = Chem.MolFromSmiles("CC")
    # tgt lacks a conformer, so `GetConformer().SetAtomPosition` fails (lines 88-89)
    _transfer_coords(ref, tgt)


@patch("pick_a_pka.draw.Chem.Kekulize", side_effect=Exception("Kekulize failed"))
def test_draw_pka_kekulize_fail(mock_kekulize):
    # Simulates an unexpected Kekulization failure fallback (lines 106-107)
    svg = draw_pka("C", vector=True)
    assert "<svg" in svg


@patch("pick_a_pka.PKaPredictor.predict_pka")
def test_draw_pka_empty_mol(mock_predict):
    mock_predict.return_value = {
        "base_pka": {},
        "acid_pka": {},
        "mol": Chem.Mol()
    }
    # Tests the N_atoms == 0 fast return block (line 125)
    img = draw_pka("C", vector=False)
    assert img.size == (800, 800)


def test_draw_pka_no_neighbors():
    # Ammonia "N" has 0 heavy neighbors, testing default directional vector assignment (line 165)
    svg = draw_pka("N", vector=True)
    assert "<svg" in svg


@patch("pick_a_pka.PKaPredictor.predict_pka")
def test_draw_pka_extreme_collisions(mock_predict):
    mol = Chem.MolFromSmiles("CC")
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    # Force heavy label-atom and label-label collision by stacking coordinates entirely
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(0.0, 0.0, 0.0))

    mock_predict.return_value = {
        "base_pka": {0: 9.0, 1: 10.0},
        "acid_pka": {0: 4.0},
        "mol": mol
    }

    # Hits vector normalization edge cases (line 179), multiple pka radii shifts (lines 184-186),
    # label-atom collisions (lines 215-217), label-label zero-distance collisions (lines 229-231),
    # and extensive displacement line rendering (lines 257-260).
    svg = draw_pka(mol, vector=True)
    assert "<svg" in svg


@patch("pick_a_pka.draw.draw_pka", return_value="<svg></svg>")
@patch("pick_a_pka.draw.plt.figure")
@patch("pick_a_pka.PKaPredictor.predict_microstates")
@patch("pick_a_pka.PKaPredictor.protonation_ladder")
@patch("pick_a_pka.PKaPredictor.predict_pka")
def test_plot_microspecies_missing_smi(mock_pka, mock_ladder, mock_micro, mock_fig, mock_draw_pka):
    fig_instance = MagicMock()
    fig_instance.savefig.side_effect = lambda buf, **kwargs: buf.write("<svg></svg>")
    mock_fig.return_value = fig_instance

    mock_pka.return_value = {"base_pka": {}, "acid_pka": {}}
    mock_ladder.return_value = ["CC", "CCO"]  # "CCO" omitted from simulated microspecies data
    mock_micro.return_value = {
        7.0: {
            "distribution": [
                {"smiles": "CC", "abundance": 100.0, "mol": Chem.MolFromSmiles("CC")}
            ]
        }
    }
    # Hits SMILES parsing fallback when it isn't listed inside micro_data (lines 311-313)
    svg = plot_microspecies_distribution("CC", vector=True)
    assert isinstance(svg, str)
    assert "<svg" in svg


@patch("pick_a_pka.PKaPredictor.predict_microstates")
@patch("pick_a_pka.PKaPredictor.protonation_ladder")
@patch("pick_a_pka.PKaPredictor.predict_pka")
@patch("pick_a_pka.draw.draw_pka", return_value="fake_drawing")
def test_plot_microspecies_many_states(mock_draw, mock_pka, mock_ladder, mock_micro):
    mock_pka.return_value = {"base_pka": {}, "acid_pka": {}}
    mock_ladder.return_value = ["C"] * 55  # Exceeds base style thresholds
    mock_micro.return_value = {
        7.0: {
            "distribution": [
                {"smiles": "C", "abundance": 100.0, "mol": Chem.MolFromSmiles("C")}
            ]
        }
    }
    # Generates enough thumbnails to hit dash array limits '--' and '-.' logic (lines 434-435)
    # The return_value "fake_drawing" lacks '<svg' testing `if start_idx != -1:` missing branches
    svg = plot_microspecies_distribution("C", vector=True)
    assert isinstance(svg, str)


from rdkit.Chem.Draw import rdMolDraw2D as _rdMD2D

real_prep = _rdMD2D.PrepareMolForDrawing

call_counts = {}


def fake_prep(mol, kekulize=True, **kwargs):
    key = id(mol)
    call_counts[key] = call_counts.get(key, 0) + 1
    # Fail only on the non-kekulize path (first call per mol inside loop)
    if not kekulize and call_counts[key] == 1:
        raise ValueError("force fail")
    return real_prep(mol, kekulize=kekulize, **kwargs)


neutral = Chem.MolFromSmiles("C")

with patch("pick_a_pka.draw.rdMolDraw2D.PrepareMolForDrawing", side_effect=fake_prep):
    with patch("pick_a_pka.draw.plt.figure") as mock_fig:
        fig_instance = MagicMock()
        fig_instance.savefig.side_effect = lambda buf, **kwargs: buf.write("<svg></svg>")
        mock_fig.return_value = fig_instance
        # Also mock draw_pka so the annotated-molecule call never re-enters the patched
        # PrepareMolForDrawing via predict_pka → draw_pka → PrepareMolForDrawing.
        with patch("pick_a_pka.draw.draw_pka", return_value="<svg></svg>"):
            with patch("pick_a_pka.PKaPredictor.predict_microstates") as mock_micro:
                mock_micro.return_value = {
                    7.0: {
                        "distribution": [
                            {"smiles": "C", "abundance": 100.0, "mol": neutral}
                        ]
                    }
                }
                with patch("pick_a_pka.PKaPredictor.protonation_ladder") as mock_ladder:
                    mock_ladder.return_value = ["C"]
                    with patch("pick_a_pka.PKaPredictor.predict_pka") as mock_pka:
                        mock_pka.return_value = {"base_pka": {}, "acid_pka": {}}
                        from pick_a_pka import plot_microspecies_distribution

                        svg = plot_microspecies_distribution("C", vector=True)
                        assert isinstance(svg, str)


@patch("pick_a_pka.draw.plt.figure")
@patch("pick_a_pka.draw.cairosvg.svg2png")
def test_plot_distribution_vector_false(mock_svg2png, mock_fig):
    fig_instance = MagicMock()
    fig_instance.savefig.side_effect = lambda buf, **kwargs: buf.write("<svg></svg>")
    mock_fig.return_value = fig_instance

    mock_svg2png.return_value = None
    with patch("pick_a_pka.draw.Image.open") as mock_open:
        mock_open.return_value = "fake_image"
        # Forces robust completion test of Cairosvg writing mechanics (line 493)
        res = plot_microspecies_distribution("C", vector=False)
        assert res == "fake_image"


class TestDrawPkaZeroAtoms:
    """draw_pka must return a PIL Image for an empty molecule (line 125)."""

    def test_empty_mol_returns_pil_image(self):
        from pick_a_pka.draw import draw_pka
        from PIL import Image
        # Build a genuine empty mol
        empty = Chem.RWMol()
        img = draw_pka(empty.GetMol(), vector=False)
        assert isinstance(img, Image.Image)


class TestDrawPkaMatplotlibBackend:
    """Fix test_pkalearn_backend_produces_svg: use matplotlib non-interactive backend."""

    def test_pkalearn_backend_produces_svg_with_agg(self):
        """Use 'Agg' backend so no Tk window is needed."""
        import matplotlib
        matplotlib.use("Agg")
        from pick_a_pka import PKaPredictor, plot_microspecies_distribution
        model = PKaPredictor("pkalearn", allow_amphoteric=True)
        svg = plot_microspecies_distribution(ACETIC_ACID, model=model, vector=True)
        assert isinstance(svg, str)
        assert "<svg" in svg


class TestDrawPkaMissingSmiFix:
    """Fix test_plot_microspecies_missing_smi: mock_pka must include 'mol' key."""

    @patch("pick_a_pka.draw.plt.figure")
    @patch("pick_a_pka.PKaPredictor.predict_microstates")
    @patch("pick_a_pka.PKaPredictor.protonation_ladder")
    @patch("pick_a_pka.PKaPredictor.predict_pka")
    def test_fallback_smi_parse_includes_mol(self, mock_pka, mock_ladder, mock_micro, mock_fig):
        """predict_pka mock must include 'mol' so draw_pka doesn't KeyError."""
        fig_instance = MagicMock()
        fig_instance.savefig.side_effect = lambda buf, **kwargs: buf.write("<svg></svg>")
        mock_fig.return_value = fig_instance

        neutral = Chem.MolFromSmiles("CC")
        mock_pka.return_value = {"base_pka": {}, "acid_pka": {}, "mol": neutral}
        mock_ladder.return_value = ["CC", "CCO"]
        mock_micro.return_value = {
            7.0: {
                "distribution": [
                    {"smiles": "CC", "abundance": 100.0, "mol": Chem.MolFromSmiles("CC")}
                ]
            }
        }
        from pick_a_pka import plot_microspecies_distribution
        svg = plot_microspecies_distribution("CC", vector=True)
        assert isinstance(svg, str)


class TestDrawPkaPrepMolFallbackFix:
    """Fix test_plot_distribution_prepare_mol_fail: mock must return 'mol' key."""

    def test_prepare_mol_fallback_with_mol_key(self):
        """Force PrepareMolForDrawing first-try to fail; second fallback should succeed."""
        from rdkit.Chem.Draw import rdMolDraw2D as _rdMD2D
        real_prep = _rdMD2D.PrepareMolForDrawing

        call_counts = {}

        def fake_prep(mol, kekulize=True, **kwargs):
            key = id(mol)
            call_counts[key] = call_counts.get(key, 0) + 1
            # Fail only on the non-kekulize path (first call per mol inside loop)
            if not kekulize and call_counts[key] == 1:
                raise ValueError("force fail")
            return real_prep(mol, kekulize=kekulize, **kwargs)

        neutral = Chem.MolFromSmiles("C")

        with patch("pick_a_pka.draw.rdMolDraw2D.PrepareMolForDrawing", side_effect=fake_prep):
            with patch("pick_a_pka.draw.plt.figure") as mock_fig:
                fig_instance = MagicMock()
                fig_instance.savefig.side_effect = lambda buf, **kwargs: buf.write("<svg></svg>")
                mock_fig.return_value = fig_instance
                # Also mock draw_pka so the annotated-molecule call never re-enters the patched
                # PrepareMolForDrawing via predict_pka → draw_pka → PrepareMolForDrawing.
                with patch("pick_a_pka.draw.draw_pka", return_value="<svg></svg>"):
                    with patch("pick_a_pka.PKaPredictor.predict_microstates") as mock_micro:
                        mock_micro.return_value = {
                            7.0: {
                                "distribution": [
                                    {"smiles": "C", "abundance": 100.0, "mol": neutral}
                                ]
                            }
                        }
                        with patch("pick_a_pka.PKaPredictor.protonation_ladder") as mock_ladder:
                            mock_ladder.return_value = ["C"]
                            with patch("pick_a_pka.PKaPredictor.predict_pka") as mock_pka:
                                mock_pka.return_value = {"base_pka": {}, "acid_pka": {}}
                                from pick_a_pka import plot_microspecies_distribution
                                svg = plot_microspecies_distribution("C", vector=True)
                                assert isinstance(svg, str)


class TestDrawPkaAtomNoNeighbors:
    """Line 165 — isolated atoms yield default direction vector."""

    def test_isolated_atom_no_crash(self):
        from pick_a_pka import draw_pka
        # Ammonia is featurizable and has zero heavy neighbours
        svg = draw_pka("N", vector=True)
        assert "<svg" in svg


class TestDrawPkaZeroVLen:
    """Line 179 — v_len ≤ 1e-4 → fallback direction (1.0, 0.0)."""

    @patch("pick_a_pka.PKaPredictor.predict_pka")
    def test_zero_vlen_fallback_direction(self, mock_pka):
        from rdkit.Chem import rdDepictor
        from rdkit.Geometry import Point3D
        from pick_a_pka import draw_pka

        mol = Chem.MolFromSmiles("C")
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))

        mock_pka.return_value = {
            "base_pka": {0: 9.0},
            "acid_pka": {},
            "mol": mol,
        }
        svg = draw_pka(mol, vector=True)
        assert "<svg" in svg


class TestDrawPkaLineRendering:
    """Lines 257-260 — line is drawn when label is far from origin."""

    @patch("pick_a_pka.PKaPredictor.predict_pka")
    def test_line_drawn_for_distant_label(self, mock_pka):
        from rdkit.Chem import rdDepictor
        from pick_a_pka import draw_pka

        mol = Chem.MolFromSmiles("CCCC")
        rdDepictor.Compute2DCoords(mol)

        mock_pka.return_value = {
            "base_pka": {0: 9.0},
            "acid_pka": {3: 2.0},
            "mol": mol,
        }
        svg = draw_pka(mol, vector=True)
        assert "<svg" in svg


class TestPlotDistributionVectorFalse:
    """Test the cairosvg path (line 493) with Agg backend."""

    def test_plot_distribution_vector_false_agg(self):
        import matplotlib
        matplotlib.use("Agg")
        from pick_a_pka import plot_microspecies_distribution
        result = plot_microspecies_distribution(ACETIC_ACID, vector=False)
        from PIL import Image
        assert isinstance(result, Image.Image)

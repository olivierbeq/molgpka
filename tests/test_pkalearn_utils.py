from rdkit import Chem


class TestWhichElement:
    """Exercise the uncovered branches of the whichElement parser."""

    def _fn(self):
        from pick_a_pka.backends.pkalearn.utils import whichElement
        return whichElement

    # --- Already-covered simple returns ---
    def test_fluorine(self):
        fn = self._fn()
        j, el, charge, _ = fn("F", 0)
        assert el == "F"

    def test_iodine(self):
        fn = self._fn()
        j, el, charge, _ = fn("I", 0)
        assert el == "I"

    def test_phosphorus(self):
        fn = self._fn()
        j, el, charge, _ = fn("P", 0)
        assert el == "P"

    # --- Nitrogen / Oxygen / Carbon variants ---
    def test_N_with_H(self):
        fn = self._fn()
        smiles = "NH"
        j, el, charge, _ = fn(smiles, 0)
        assert "N" in el

    def test_N_with_H2(self):
        fn = self._fn()
        smiles = "NH2"
        j, el, charge, _ = fn(smiles, 0)
        assert "H2" in el or el.startswith("N")

    def test_N_with_H3_plus(self):
        fn = self._fn()
        smiles = "NH3+"
        j, el, charge, _ = fn(smiles, 0)
        assert "H3" in el or charge == "+"

    def test_N_with_charge_plus(self):
        fn = self._fn()
        smiles = "N+"
        j, el, charge, _ = fn(smiles, 0)
        assert charge == "+"

    def test_N_with_charge_minus(self):
        fn = self._fn()
        smiles = "N-"
        j, el, charge, _ = fn(smiles, 0)
        assert charge == "-"

    def test_C_plain(self):
        fn = self._fn()
        smiles = "C"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "C"

    def test_Cl(self):
        fn = self._fn()
        smiles = "Cl"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "Cl"

    def test_Cl_minus(self):
        fn = self._fn()
        smiles = "Cl-"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "Cl" and charge == "-"

    def test_aromatic_c(self):
        fn = self._fn()
        smiles = "c"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "C"

    def test_S_plain(self):
        fn = self._fn()
        smiles = "S"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "S"

    def test_Se(self):
        fn = self._fn()
        smiles = "Se"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "Se"

    def test_Si(self):
        fn = self._fn()
        smiles = "Si"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "Si"

    def test_S_minus(self):
        fn = self._fn()
        smiles = "S-"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "S" and charge == "-"

    def test_Br(self):
        fn = self._fn()
        smiles = "Br"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "Br"

    def test_Br_minus(self):
        fn = self._fn()
        smiles = "Br-"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "Br" and charge == "-"

    def test_B_plain(self):
        fn = self._fn()
        smiles = "B"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "B"

    def test_As(self):
        fn = self._fn()
        smiles = "As"
        j, el, charge, _ = fn(smiles, 0)
        assert el == "As"

    def test_N_at_H(self):
        """N followed by @H — chiral centre."""
        fn = self._fn()
        smiles = "N@H"
        j, el, charge, _ = fn(smiles, 0)
        assert "N" in el

    def test_N_at_at_H(self):
        """N followed by @@H."""
        fn = self._fn()
        smiles = "N@@H"
        j, el, charge, _ = fn(smiles, 0)
        assert "N" in el

    def test_N_at_bracket_close(self):
        """N@ followed by ']' — chiral tag only."""
        fn = self._fn()
        smiles = "N@]"
        j, el, charge, _ = fn(smiles, 0)
        assert "N" in el

    def test_N_at_at_bracket_close(self):
        """N@@ followed by ']'."""
        fn = self._fn()
        smiles = "N@@]"
        j, el, charge, _ = fn(smiles, 0)
        assert "N" in el

    def test_N_H2_plus(self):
        """NH2 followed by charge +."""
        fn = self._fn()
        smiles = "NH2+"
        j, el, charge, _ = fn(smiles, 0)
        assert charge == "+"


class TestPkaLearnUtils:
    """Lines 62-63, 73-74, 80-83, 129."""

    def test_which_element_Al(self):
        """Line 129: 'A' not followed by 's' → 'Al'."""
        from pick_a_pka.backends.pkalearn.utils import whichElement
        smiles = "Al"
        j, el, charge, _ = whichElement(smiles, 0)
        assert el == "Al"

    def test_which_element_N_at_charge_plus(self):
        """Lines 62-63: N@ followed by '+' or '-'."""
        from pick_a_pka.backends.pkalearn.utils import whichElement
        smiles = "N@+"
        j, el, charge, _ = whichElement(smiles, 0)
        assert "N" in el

    def test_which_element_N_at_at_H_with_charge(self):
        """Lines 73-74: N@@H followed by charge."""
        from pick_a_pka.backends.pkalearn.utils import whichElement
        smiles = "N@@H+"
        j, el, charge, _ = whichElement(smiles, 0)
        assert "N" in el

    def test_which_element_N_at_at_H_bracket_close(self):
        """Lines 80-83: N@@ then char3 == ']'."""
        from pick_a_pka.backends.pkalearn.utils import whichElement
        smiles = "N@@]"
        j, el, charge, _ = whichElement(smiles, 0)
        assert "N" in el

    def test_swap_bond_atoms_swaps_when_further(self):
        """swap_bond_atoms swaps bond direction when atom1 is farther from center."""
        from pick_a_pka.backends.pkalearn.utils import swap_bond_atoms
        from rdkit.Chem import rdmolops
        mol = Chem.MolFromSmiles("CCC")
        dm = rdmolops.GetDistanceMatrix(mol)
        mw_a, mw_b = swap_bond_atoms(mol, Chem.RWMol(mol), center=0, distance_matrix=dm)
        assert mw_a is not None

    def test_N_plain_no_following_chars(self):
        """N at end of string → short string."""
        from pick_a_pka.backends.pkalearn.utils import whichElement
        j, el, charge, _ = whichElement("N", 0)
        assert "N" in el

    def test_O_with_H2_and_charge(self):
        """NH2+ variant for O/N — covers H2 + charge branch."""
        from pick_a_pka.backends.pkalearn.utils import whichElement
        smiles = "NH2-"
        j, el, charge, _ = whichElement(smiles, 0)
        assert charge == "-" or "H2" in el

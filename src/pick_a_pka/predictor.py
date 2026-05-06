from typing import Literal

from rdkit import Chem

from .backends.molgpka.model import MolGpKaModel
from .backends.pkalearn.model import PkaLearnModel
from .core.base import BasePKaModel
from .core.exceptions import InvalidBackendError
from .core.types import BackendType, MicrostateResult


def _generate_ordered_states(mol_no_hs, base_pka_dict, acid_pka_dict):
    """Return a list of RDKit molecules representing the protonation states
    in ascending pKa order (most protonated → most deprotonated).

    This is the backend-agnostic implementation of the protonation ladder.
    It mirrors MolGpKa's _generate_microspecies_sequence and works for any
    backend whose predict_pka returns base_pka / acid_pka / mol dicts.
    """
    ionizable_sites = []
    for idx, pka in base_pka_dict.items():
        ionizable_sites.append((pka, idx, 'base'))
    for idx, pka in acid_pka_dict.items():
        ionizable_sites.append((pka, idx, 'acid'))
    ionizable_sites.sort(key=lambda x: x[0])

    if not ionizable_sites:
        return [mol_no_hs]

    unique_atoms = {idx for _, idx, _ in ionizable_sites}
    # For each ionizable atom, count how many basic pKa values it has —
    # that determines its charge in the fully-protonated state.
    fully_protonated_charges = {
        atom_idx: sum(1 for _, i, t in ionizable_sites if i == atom_idx and t == 'base')
        for atom_idx in unique_atoms
    }

    states = []
    for k in range(len(ionizable_sites) + 1):
        rw = Chem.RWMol(mol_no_hs)
        try:
            Chem.Kekulize(rw, clearAromaticFlags=True)
        except Exception:
            pass

        for atom_idx in unique_atoms:
            atom = rw.GetAtomWithIdx(atom_idx)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.SetFormalCharge(fully_protonated_charges[atom_idx])

        for i in range(k):
            _, idx, _ = ionizable_sites[i]
            atom = rw.GetAtomWithIdx(idx)
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)

        try:
            rw.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(rw)
            states.append(rw.GetMol())
        except Exception:
            pass

    return states


class PKaPredictor(BasePKaModel):
    def __init__(
            self,
            model: Literal["molgpka", "pkalearn"] | BackendType = BackendType.MOLGPKA,
            device: str = "cpu",
            allow_amphoteric: bool = False,
    ):
        try:
            self.model_name = BackendType(model)
        except ValueError:
            raise InvalidBackendError(
                f"Unknown backend: '{model}'. Choose from: {[b.value for b in BackendType]}"
            )
        super().__init__(device=device)
        self.allow_amphoteric = allow_amphoteric
        if self.model_name == BackendType.MOLGPKA:
            self.model = MolGpKaModel(device=self.device)
        elif self.model_name == BackendType.PKALEARN:
            self.model = PkaLearnModel(device=self.device,
                                       allow_amphoteric=self.allow_amphoteric
                                       )

    def __del__(self):
        # GPU cleanup
        if hasattr(self, "model") and hasattr(self.model, "dispose"):
            self.model.dispose()

    def predict_pka(self, mol: Chem.Mol | list[Chem.Mol] | str | list[str]) -> list[dict[int, float]]:
        """Predict the pKa values for a molecule or a list of molecules.

        :param mol: molecule, SMILES, or a list of either
        :return: a dictionary mapping each atom ID to its pKa value, for each molecule provided.
        """
        mols = self._to_mol(mol)
        results = [self.model.predict_pka(m) for m in mols]
        return results if isinstance(mol, list) else results[0]

    def predict_microstates(self, mol: Chem.Mol | list[Chem.Mol] | str | list[str],
                            ph: float | list[float] = 7.4,
                            ph_range: tuple = None, ph_step: float = None
                            ) -> list[MicrostateResult | dict[float, MicrostateResult]]:
        """Predict the relative abundances of the microstates for a molecule or a list of molecules at a given pH.

        :param mol: molecule, SMILES, or a list of either
        :param ph: A single pH value to determine the relative abundance of molecular micro-species at.
        :param ph_range: A range of pH to determine the relative abundance of molecular micro-species at. Ignored if `ph` is not None.
        :param ph_step: The incremental step to consider between values of the `ph_range`. Ignored if ph_range is None.
        :return: A list of `MicrostateResult` (if pH is not None) or of dictionaries for each molecule provided.
        """
        mols = self._to_mol(mol)
        results = [
            self.model.predict_microstates(mol_, ph=ph, ph_range=ph_range, ph_step=ph_step)
            for mol_ in mols
        ]
        return results if isinstance(mol, list) else results[0]

    def protonation_ladder(
            self,
            mol: Chem.Mol | str,
            acid_first: bool = True,
    ) -> list[str]:
        """Return the protonation states of a molecule as a list of canonical SMILES,
        ordered along the deprotonation ladder.

        The ladder is derived from :meth:`predict_pka` and is therefore
        backend-agnostic: it works identically for both ``molgpka`` and
        ``pkalearn``.

        :param mol: molecule or SMILES string.
        :param acid_first: if ``True`` (default), the list runs from the most
            protonated state (lowest pH / highest charge) to the most
            deprotonated state.  Set to ``False`` to reverse the order
            (most deprotonated first).
        :return: list of canonical SMILES strings, one per protonation state,
            in the requested order.  Always contains at least one entry (the
            neutral input molecule) even for non-ionisable structures.
        """
        input_mol = self._to_mol(mol)[0]
        pred = self.predict_pka(input_mol)
        mol_no_hs = pred["mol"]
        base_pka_dict = pred["base_pka"]
        acid_pka_dict = pred["acid_pka"]

        states = _generate_ordered_states(mol_no_hs, base_pka_dict, acid_pka_dict)

        # Deduplicate while preserving order (identical SMILES can arise when
        # two ionizable atoms have identical pKa values and RDKit collapses them).
        seen = set()
        smiles_list = []
        for state_mol in states:
            smi = Chem.MolToSmiles(state_mol)
            if smi not in seen:
                seen.add(smi)
                smiles_list.append(smi)

        if not acid_first:
            smiles_list = list(reversed(smiles_list))

        return smiles_list

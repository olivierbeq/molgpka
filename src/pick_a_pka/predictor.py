from typing import Literal

from rdkit import Chem

from .backends.molgpka.model import MolGpKaModel
from .backends.pkalearn.model import PkaLearnModel


class PKaPredictor:
    def __init__(
            self,
            model: Literal["pkalearn", "molgpka"] = "pkalearn",
            device: str = "cpu",
    ):
        if model not in ["molgpka", "pkalearn"]:
            raise ValueError(f"Unknown backend: {model}. Choose 'molgpka' or 'pkalearn'.")

        self.model_name = model
        self.device = device

        if self.model_name == "molgpka":
            self.model = MolGpKaModel(device=self.device)
        elif self.model_name == "pkalearn":
            self.model = PkaLearnModel(device=self.device)

    def __del__(self):
        # GPU cleanup
        if hasattr(self.model, "dispose"):
            self.model.dispose()

    def _to_mol(self, mol_or_smiles: Chem.Mol | str | list[Chem.Mol] | list[str]) -> list[Chem.Mol]:
        """Parse a molecule or list of molecules.

        :param mol_or_smiles: molecule, SMILES, or a list of either
        :return: a list of RDKit molecule object(s)
        """
        if isinstance(mol_or_smiles, list):
            return sum([self._to_mol(mol) for mol in mol_or_smiles])
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            return [mol]
        return [mol_or_smiles]

    def predict_pka(self, mol: Chem.Mol | list[Chem.Mol]) -> list[dict[int, float]]:
        """Predict the pKa values for a molecule or a list of molecules.

        :param mol: molecule, SMILES, or a list of either
        :return: a dictionary mapping each atom ID to its pKa value, for each molecule provided.
        """
        if isinstance(mol, list):
            return sum([self.predict_pka(mol_) for mol_ in mol])
        else:
            mol = self._to_mol(mol)
        return [self.model.predict_pka(mol_) for mol_ in mol]

    def predict_microstates(self, mol: Chem.Mol | list[Chem.Mol], ph: float | list[float] = 7.4,
                            ph_range: tuple = None, ph_step: float = None
                            ) -> list[dict[float, Chem.Mol] | dict[float, dict[float, Chem.Mol]]]:
        """Predict the relative abundances of the microstates for a molecule or a list of molecules at a given pH.

        :param mol: molecule, SMILES, or a list of either
        :param ph: A single pH value to determine the relative abundance of molecular micro-species at.
        :param ph_range: A range of pH to determine the relative abundance of molecular micro-species at. Ignored if `ph` is not None.
        :param ph_step: The incremental step to consider between values of the `ph_range`. Ignored if ph_range is None.
        :return: A list of dictionaries for each molecule provided.
        If `ph` is not None, each dictionary contains relative abundances as keys and micro-species molecules as values {relative_abundance_float: Chem.Mol}.
        If `ph_range` is not None, each dictionary contains incremental pH steps as keys and dictionaries as values,
        each with relative abundances as keys and corresponding micro-species molecules as values {pH_float: {relative_abundance_float: Chem.Mol}}.
        """
        if isinstance(mol, list):
            return [self.predict_microstates(mol_, ph=ph, ph_range=ph_range, ph_step=ph_step) for mol_ in mol]
        mol = self._to_mol(mol)
        return self.model.predict_microstates(mol, ph=ph, ph_range=ph_range, ph_step=ph_step)

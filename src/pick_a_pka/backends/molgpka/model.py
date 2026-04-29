import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem
from importlib import resources

from pick_a_pka.core.base import BasePKaModel
from .network import MolGpKaNet
from .featurizer import mol_to_graph, get_ionization_aid
from .protonation import compute_microstates


class MolGpKaModel(BasePKaModel):
    def __init__(self, device="cpu"):
        super().__init__(device=device)
        # Load weights from the dedicated resources namespace
        pkg = "pick_a_pka.backends.molgpka.resources"
        self.model_acid = self._load_model(pkg, 'weight_acid.pth')
        self.model_base = self._load_model(pkg, 'weight_base.pth')

    def _load_model(self, pkg, filename):
        model = MolGpKaNet()
        with resources.as_file(resources.files(pkg).joinpath(filename)) as path:
            state = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_pka(self, mol, uncharged=True):
        """
        Featurize and infer pKa values for acidic and basic sites on the given RDKit molecule.
        """
        mol_copy = Chem.Mol(mol)
        if uncharged:
            un = rdMolStandardize.Uncharger()
            mol_uncharged = un.uncharge(mol_copy)
            mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol_uncharged))
        else:
            mol_clean = mol_copy

        mol_h = AllChem.AddHs(mol_clean)

        base_pka = self._predict_base(mol_h)
        acid_pka = self._predict_acid(mol_h)

        # Map indices back to the Hydrogen-depleted molecule
        mol_no_hs, base_dict, acid_dict = self._remap_pka_without_hs(mol_h, base_pka, acid_pka)

        return {
            "base_pka": base_dict,
            "acid_pka": acid_dict,
            "mol": mol_no_hs
        }

    def _predict_acid(self, mol):
        acid_idxs = get_ionization_aid(mol, acid_or_base="acid")
        acid_res = {}
        for aid in acid_idxs:
            data = mol_to_graph(mol, aid).to(self.device)
            out = self.model_acid(data)
            acid_res[aid] = out.item()
        return acid_res

    def _predict_base(self, mol):
        base_idxs = get_ionization_aid(mol, acid_or_base="base")
        base_res = {}
        for aid in base_idxs:
            data = mol_to_graph(mol, aid).to(self.device)
            out = self.model_base(data)
            base_res[aid] = out.item()
        return base_res

    def predict_microstates(self, mol, pH=7.4):
        """Returns major protonation state and distribution at a given pH."""
        return compute_microstates(self, mol, pH)

    def _remap_pka_without_hs(self, mol_with_hs, base_pka_dict, acid_pka_dict):
        """
        Remap pKa atom indices in a molecule with explicit hydrogens to the molecule without hydrogens.
        """
        for atom in mol_with_hs.GetAtoms():
            atom.SetIntProp("OrigIdx", atom.GetIdx())

        h_to_heavy = {}
        for atom in mol_with_hs.GetAtoms():
            if atom.GetAtomicNum() == 1:
                neighbors = atom.GetNeighbors()
                if neighbors:
                    h_to_heavy[atom.GetIdx()] = neighbors[0].GetIdx()

        mol_no_hs = Chem.RemoveHs(mol_with_hs)

        orig_to_new_idx = {}
        for atom in mol_no_hs.GetAtoms():
            if atom.HasProp("OrigIdx"):
                orig_to_new_idx[atom.GetIntProp("OrigIdx")] = atom.GetIdx()

        new_acid_pka_dict = {}
        new_base_pka_dict = {}

        for old_idx, pka_val in acid_pka_dict.items():
            if old_idx in orig_to_new_idx:
                new_acid_pka_dict[orig_to_new_idx[old_idx]] = pka_val
            elif old_idx in h_to_heavy and h_to_heavy[old_idx] in orig_to_new_idx:
                new_acid_pka_dict[orig_to_new_idx[h_to_heavy[old_idx]]] = pka_val

        for old_idx, pka_val in base_pka_dict.items():
            if old_idx in orig_to_new_idx:
                new_base_pka_dict[orig_to_new_idx[old_idx]] = pka_val
            elif old_idx in h_to_heavy and h_to_heavy[old_idx] in orig_to_new_idx:
                new_base_pka_dict[orig_to_new_idx[h_to_heavy[old_idx]]] = pka_val

        return mol_no_hs, new_base_pka_dict, new_acid_pka_dict

    def dispose(self):
        del self.model_acid
        del self.model_base
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

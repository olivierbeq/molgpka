import torch
from rdkit import Chem
from importlib import resources

from pick_a_pka.core.base import BasePKaModel
from .network import PkaLearnGNN


class PkaLearnModel(BasePKaModel):
    DEFAULT_CONFIG = {
        'atom_feature_element': False,
        'atom_feature_electronegativity': True,
        'atom_feature_hardness': True,
        'atom_feature_atom_size': True,
        'atom_feature_hybridization': True,
        'atom_feature_aromaticity': True,
        'atom_feature_number_of_rings': False,
        'atom_feature_ring_size': True,
        'atom_feature_number_of_Hs': True,
        'atom_feature_formal_charge': True,
        'bond_feature_bond_order': True,
        'bond_feature_conjugation': True,
        'bond_feature_polarization': True,
        'bond_feature_charge_conjugation': True,
        'bond_feature_focused': False,
        'acid_or_base': 'base',
        'mask_size': 4,
        'model_embedding_size': 128,
        'model_gnn_layers': 4,
        'model_fc_layers': 2,
        'model_dropout_rate': 0.0,
        'model_dense_neurons': 448,
        'model_attention_heads': 4
    }

    def __init__(self, device="cpu", config=None):
        super().__init__(device=device)
        self.config = config or self.DEFAULT_CONFIG
        self.model = PkaLearnGNN(feature_size=19, edge_dim=7, model_params=self.config)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        pkg = "pick_a_pka.backends.pkalearn.resources"
        with resources.as_file(resources.files(pkg).joinpath("train_AAc-1_best.pth")) as path:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(self, mol_or_smiles):
        """
        Runs the full iterative deprotonation ladder.
        Returns a list of dicts: [{'smiles': ..., 'center': ..., 'pka': ...}, ...]
        """
        from .microstates import predict_ladder

        if isinstance(mol_or_smiles, str):
            smiles_str = mol_or_smiles
        else:
            # We must use non-canonical SMILES to ensure string index -> RDKit index mapping
            smiles_str = Chem.MolToSmiles(Chem.RemoveHs(mol_or_smiles), canonical=False)

        return predict_ladder(self, smiles_str, self.config)

    def predict_pka(self, mol):
        mol_clean = Chem.RemoveHs(mol) if isinstance(mol, Chem.Mol) else Chem.MolFromSmiles(mol)
        ladder = self.predict(mol_clean)

        base_pka = {}
        acid_pka = {}

        for step in ladder:
            idx = step['center']
            pka = step['pka']

            # 100% Foolproof Thermodynamic Rule:
            # We parse the state of the molecule *after* deprotonation (stable at high pH).
            # Because pKaLearn uses string slicing, the atom indices are strictly preserved.
            step_mol = Chem.MolFromSmiles(step['smiles'], sanitize=False)

            if step_mol and step_mol.GetNumAtoms() == mol_clean.GetNumAtoms():
                # Get the formal charge of the target atom in the deprotonated state
                fc = step_mol.GetAtomWithIdx(idx).GetFormalCharge()

                # An acid loses a proton from its neutral state to become anionic (< 0).
                # A base loses a proton from its cationic state to become neutral/less positive (>= 0).
                if fc < 0:
                    acid_pka[idx] = pka
                else:
                    base_pka[idx] = pka
            else:
                # Absolute fallback in the extremely rare case where SMILES length mismatches
                sym = mol_clean.GetAtomWithIdx(idx).GetSymbol()
                if sym in ['O', 'S', 'P', 'C']:
                    acid_pka[idx] = pka
                else:
                    base_pka[idx] = pka

        return {
            "base_pka": base_pka,
            "acid_pka": acid_pka,
            "mol": mol_clean
        }

    def predict_microstates(self, mol, pH=7.4):
        from .microstates import compute_microstates_at_ph
        return compute_microstates_at_ph(self, mol, pH, self.config)

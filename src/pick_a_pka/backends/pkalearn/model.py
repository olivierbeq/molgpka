import copy
from importlib import resources

import torch
from rdkit import Chem

from .network import PkaLearnGNN
from ...core.base import BasePKaModel
from ...core.exceptions import ResourceNotFoundError


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

    def __init__(self, device="cpu", config=None, allow_amphoteric: bool = False):
        super().__init__(device=device)
        self.config = config or self.DEFAULT_CONFIG
        self.allow_amphoteric = allow_amphoteric
        self.model = PkaLearnGNN(feature_size=19, edge_dim=7, model_params=self.config)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        pkg = "pick_a_pka.backends.pkalearn.resources"
        try:
            with resources.as_file(resources.files(pkg).joinpath("train_AAc-1_best.pth")) as path:
                ckpt = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise ResourceNotFoundError(f"Cound not load pKaLearn model weights: {e}")

    @torch.no_grad()
    def predict(self, mol_or_smiles):
        """
        Runs the full iterative deprotonation ladder.

        Returns a tuple (ladder, starting_mol) where:
          - ladder is a list of LadderStep dicts: [{'smiles', 'center', 'pka'}, ...]
          - starting_mol is the RDKit Mol actually fed to the ladder (pre-protonated
            when allow_amphoteric=True).
        """
        from .microstates import predict_ladder

        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = Chem.Mol(mol_or_smiles)

        mol_clean = Chem.RemoveHs(mol)

        if self.allow_amphoteric:
            # Pre-protonate all neutral nitrogens so they enter the ladder at the top.
            # Only aliphatic nitrogens with degree <= 3 are touched; aromatic ones are
            # left unchanged (SetFormalCharge on an aromatic n breaks kekulization).
            rw_mol = Chem.RWMol(mol_clean)
            patt = Chem.MolFromSmarts('[#7+0]')
            if patt:
                for m in rw_mol.GetSubstructMatches(patt):
                    atom = rw_mol.GetAtomWithIdx(m[0])
                    if atom.GetDegree() <= 3:
                        atom.SetFormalCharge(1)
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            try:
                Chem.SanitizeMol(rw_mol)
                mol_clean = rw_mol.GetMol()
            except Exception:
                mol_clean = Chem.RemoveHs(mol)

        # Non-canonical SMILES preserves the native node order perfectly.
        smiles_str = Chem.MolToSmiles(mol_clean, canonical=False)
        # The ladder runs with allow_amphoteric=False: amphoteric second-pass inference
        # is handled entirely in predict_pka, not inside the ladder, to avoid duplicate
        # steps for the same atom.
        ladder = predict_ladder(self, smiles_str, self.config, allow_amphoteric=False)
        return ladder, mol_clean

    def predict_pka(self, mol):
        from .featurizer import from_acid_to_base
        from .featurizer import mol_to_graph
        from .inference import predict_single

        neutral_mol = Chem.RemoveHs(mol) if isinstance(mol, Chem.Mol) else Chem.MolFromSmiles(mol)
        ladder, _ = self.predict(Chem.Mol(neutral_mol))

        base_pka = {}
        acid_pka = {}

        for step in ladder:
            pka = step['pka']
            idx = step['center']

            if idx >= neutral_mol.GetNumAtoms():
                continue

            # step['smiles'] is the protonated (acid-form) SMILES the ladder was
            # evaluating. Apply from_acid_to_base to simulate removing one proton,
            # then check the resulting fc at the center atom:
            #   fc < 0  -> deprotonation produced an anion  -> acidic pKa
            #   fc >= 0 -> deprotonation produced a neutral -> basic pKa
            step_mol = Chem.MolFromSmiles(step['smiles'])
            if step_mol is None:
                step_mol = Chem.MolFromSmiles(step['smiles'], sanitize=False)
            if step_mol is None or idx >= step_mol.GetNumAtoms():
                continue

            deprotonated = copy.deepcopy(step_mol)
            b_found, deprotonated, _ = from_acid_to_base(deprotonated, idx)
            if not b_found:
                continue

            post_fc = deprotonated.GetAtomWithIdx(idx).GetFormalCharge()
            if post_fc < 0:
                # Only record as acidic if this atom was not already seen as basic —
                # a duplicate acid entry for a basic atom means the ladder visited it
                # twice (once per ionization state), and the amphoteric extension below
                # will provide the correct acid pKa from the neutral context instead.
                if idx not in base_pka:
                    acid_pka[idx] = pka
            else:
                base_pka[idx] = pka

        # --- Amphoteric extension ---
        # Atoms classified as basic (e.g. NH3+ -> NH2 in the ladder) still carry a
        # proton in their *neutral* form and can also act as acids (NH2 -> NH-).
        # The ladder never reaches that second deprotonation (it advanced after the
        # basic step). When allow_amphoteric=True we run one extra inference pass per
        # such atom, using the neutral molecule as the GNN input context.
        if self.allow_amphoteric and base_pka:
            for idx in list(base_pka.keys()):
                if idx in acid_pka:
                    continue

                atom = neutral_mol.GetAtomWithIdx(idx)

                # Must carry a proton in its neutral form to act as an acid
                if atom.GetTotalNumHs() == 0:
                    continue
                # Only meaningful for classic amphoteric heteroatoms
                if atom.GetSymbol() not in ('N', 'O', 'S', 'P'):
                    continue

                # Confirm deprotonation of the neutral atom yields an anion
                mol_check = copy.deepcopy(neutral_mol)
                b_found, mol_check, _ = from_acid_to_base(mol_check, idx)
                if not b_found:
                    continue
                if mol_check.GetAtomWithIdx(idx).GetFormalCharge() >= 0:
                    continue  # no anion formed -> not a genuine acidic site

                data = mol_to_graph(copy.deepcopy(neutral_mol), idx, self.config)
                if data is None:
                    continue

                acid_pka[idx] = predict_single(self.model, data, self.device)

        return {
            "base_pka": base_pka,
            "acid_pka": acid_pka,
            "mol": neutral_mol
        }

    def predict_microstates(self, mol, ph=7.4, ph_range=None, ph_step=None):
        from .microstates import compute_microstates
        return compute_microstates(self, mol, ph=ph, ph_range=ph_range, ph_step=ph_step)

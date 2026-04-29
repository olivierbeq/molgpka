import copy
from rdkit import Chem
from .change_ionization import parse_smiles, find_centers, addHs, ionizeN
from .featurizer import mol_to_graph
from .inference import predict_single


class DummyArgs:
    def __init__(self):
        self.carbons_included = False
        self.verbose = 0
        self.mode = "infer"


def _isDigit(char):
    return char in '0123456789'


def _clean_smiles(smiles):
    j = 0
    while j < len(smiles):
        if smiles[j] == ':':
            pos = len(smiles)
            for k in range(j + 1, len(smiles)):
                if not _isDigit(smiles[k]):
                    pos = k
                    break
            smiles = smiles[:j] + smiles[pos:]
            smiles = smiles.replace('[N]', 'N').replace('[n]', 'n').replace('[O]', 'O')
            break
        j += 1
    return smiles


def _infer_round(model_wrapper, smiles, initial, ionization_states_in, config):
    dummy_args = DummyArgs()

    if initial:
        smiles = _clean_smiles(smiles)
        smiles = smiles.replace('([H])', '').replace('[H]', '').replace('[C-]', 'C').replace('-c', 'c').replace('[n]',
                                                                                                                'n'
                                                                                                                )

    mol_original = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol_original:
        Chem.rdmolops.RemoveHs(mol_original, sanitize=False)
        Chem.SanitizeMol(mol_original, catchErrors=True)

    negative_nitrogens = []
    pyridinium = []

    if initial:
        # Find sites on neutral molecule
        ionizable_nitrogens, positive_nitrogens, acidic_nitrogens, negative_oxygens, \
            acidic_oxygens, acidic_carbons, nitro_nitrogens = find_centers(mol_original, 0, smiles, "mol", initial,
                                                                           dummy_args
                                                                           )

        smiles = ionizeN(smiles, mol_original, mol_original.GetNumAtoms(), acidic_nitrogens, acidic_oxygens,
                         acidic_carbons, ionizable_nitrogens, negative_nitrogens, negative_oxygens, nitro_nitrogens,
                         pyridinium, dummy_args
                         )

        smiles = _clean_smiles(smiles)
        j = 0
        atom_idx = 0
        while j < len(smiles):
            _, smiles, j, atom_idx = parse_smiles(smiles, j, atom_idx, initial, ionizable_nitrogens,
                                                  positive_nitrogens, acidic_nitrogens, negative_nitrogens,
                                                  negative_oxygens, acidic_oxygens, acidic_carbons,
                                                  pyridinium, nitro_nitrogens, False, True
                                                  )

        mol_original = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol_original:
            Chem.rdmolops.RemoveHs(mol_original, sanitize=False)
            Chem.SanitizeMol(mol_original, catchErrors=True)
            smiles = addHs(smiles, mol_original, mol_original.GetNumAtoms(), negative_nitrogens)
            smiles = ionizeN(smiles, mol_original, mol_original.GetNumAtoms(), acidic_nitrogens, acidic_oxygens,
                             acidic_carbons, ionizable_nitrogens, negative_nitrogens, negative_oxygens, nitro_nitrogens,
                             pyridinium, dummy_args
                             )

        ionization_states0 = [
            ionizable_nitrogens, positive_nitrogens, acidic_nitrogens,
            negative_nitrogens, negative_oxygens, acidic_oxygens,
            acidic_carbons, nitro_nitrogens
        ]
    else:
        ionization_states0 = ionization_states_in

    predicts, inf_smiles_list, centers, ion_states_list = [], [], [], []
    j, atom_idx = -1, 0
    smiles_A = smiles

    while j < len(smiles_A):
        st = [copy.deepcopy(x) for x in ionization_states0]
        if j < 0: j = 0

        is_smiles, smiles_A, j, atom_idx = parse_smiles(
            smiles, j, atom_idx, initial, st[0], st[1], st[2], st[3], st[4], st[5], st[6], pyridinium, st[7], True,
            False
        )

        if is_smiles:
            mol_obj_A = Chem.MolFromSmiles(smiles_A, sanitize=False)
            if not mol_obj_A: continue
            Chem.rdmolops.RemoveHs(mol_obj_A, sanitize=False)
            Chem.SanitizeMol(mol_obj_A, catchErrors=True)

            center = atom_idx - 1
            data = mol_to_graph(mol_obj_A, center, config)
            if data is None: continue

            predicts.append(predict_single(model_wrapper.model, data, model_wrapper.device))
            inf_smiles_list.append(smiles_A)
            centers.append(center)
            ion_states_list.append(st)

    return predicts, inf_smiles_list, centers, ion_states_list


def predict_ladder(model_wrapper, original_smiles, config):
    """Iterative macroscopic deprotonation sequence."""
    all_results = []
    initial = True
    curr_smiles = original_smiles
    curr_ion_states = []

    while True:
        predicts, smis, centers, states = _infer_round(model_wrapper, curr_smiles, initial, curr_ion_states, config)
        if not predicts: break

        # Take the highest pKa (the one that stays protonated longest)
        best_idx = predicts.index(max(predicts))

        all_results.append({
            "smiles": smis[best_idx],
            "center": centers[best_idx],
            "pka": predicts[best_idx]
        }
        )

        # Prepare for next round
        curr_smiles = smis[best_idx]
        curr_ion_states = states[best_idx]
        initial = False
        if predicts[best_idx] < -10: break

    return all_results


def compute_microstates_at_ph(model_wrapper, mol, pH, config):
    ladder = predict_ladder(model_wrapper, Chem.MolToSmiles(mol, canonical=False), config)
    if not ladder: return {"major_state": mol, "pka": None, "ladder": []}

    # Simple threshold filter: take the last state where pKa > pH
    dominant = ladder[0]
    for step in ladder:
        if step["pka"] >= pH:
            dominant = step
        else:
            break

    return {
        "major_state": Chem.MolFromSmiles(dominant["smiles"]),
        "pka": dominant["pka"],
        "ladder": ladder
    }

import math
import numpy as np
from copy import deepcopy
from itertools import combinations
from rdkit import Chem


def _generate_microspecies_sequence(mol_no_hs, base_pka_dict, acid_pka_dict):
    """Generate a list of RDKit molecules representing sequential deprotonation states."""
    ionizable_sites = []
    for idx, pka in base_pka_dict.items():
        ionizable_sites.append((pka, idx, 'base'))
    for idx, pka in acid_pka_dict.items():
        ionizable_sites.append((pka, idx, 'acid'))

    ionizable_sites.sort(key=lambda x: x[0])

    unique_ionizable_atoms = set([idx for _, idx, _ in ionizable_sites])
    fully_protonated_charges = {}

    for atom_idx in unique_ionizable_atoms:
        base_count = sum(1 for _, i, site_type in ionizable_sites if i == atom_idx and site_type == 'base')
        fully_protonated_charges[atom_idx] = base_count

    microspecies_list = []

    for k in range(len(ionizable_sites) + 1):
        rw_mol = Chem.RWMol(mol_no_hs)
        Chem.Kekulize(rw_mol, clearAromaticFlags=True)

        for atom_idx in unique_ionizable_atoms:
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.SetFormalCharge(fully_protonated_charges[atom_idx])

        for i in range(k):
            _, idx, _ = ionizable_sites[i]
            atom = rw_mol.GetAtomWithIdx(idx)
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)

        rw_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(rw_mol)
        microspecies_list.append(rw_mol.GetMol())

    return microspecies_list


def calculate_microspecies_abundances(model, mol, ph=None, ph_range=None, ph_step=None):
    if ph is None and ph_range is None:
        raise ValueError("Either `ph` or `ph_range` must be specified.")
    if ph_range is not None and ph_step is None:
        raise ValueError("`ph_step` must be specified when using `ph_range`.")

    pred = model.predict(mol, uncharged=True)
    base_pka_dict, acid_pka_dict, mol_no_hs = pred["base_pka"], pred["acid_pka"], pred["mol"]
    microspecies = _generate_microspecies_sequence(mol_no_hs, base_pka_dict, acid_pka_dict)

    all_pkas = sorted(list(base_pka_dict.values()) + list(acid_pka_dict.values()))
    n_pkas = len(all_pkas)

    def get_abundances_at_ph(current_ph):
        log_ratios = [0.0]
        current_sum_pka = 0.0

        for k in range(1, n_pkas + 1):
            current_sum_pka += all_pkas[k - 1]
            log_ratios.append(k * current_ph - current_sum_pka)

        max_log = max(log_ratios)
        ratios = [10 ** (lr - max_log) for lr in log_ratios]

        total_ratio = sum(ratios)
        fractions = [(r / total_ratio) * 100.0 for r in ratios]

        result = {}
        for frac, state_mol in zip(fractions, microspecies):
            key = frac
            while key in result:
                key += 1e-12
            result[key] = state_mol
        return result

    if ph is not None:
        return get_abundances_at_ph(ph)
    else:
        results_over_range = {}
        for current_ph in np.arange(ph_range[0], ph_range[1] + (ph_step / 2), ph_step):
            rounded_ph = round(current_ph, max(0, int(math.ceil(-math.log10(ph_step)))))
            results_over_range[rounded_ph] = get_abundances_at_ph(current_ph)
        return results_over_range


def compute_microstates(model, mol, pH):
    """Wrapper for calculating dominant states at a given pH."""
    abundances = calculate_microspecies_abundances(model, mol, ph=pH)
    major_state = max(abundances.items(), key=lambda x: x[0])[1]
    return {
        "states": abundances,
        "major_state": major_state
    }


def modify_mol(mol_no_hs, acid_dict, base_dict):
    """Annotates atoms with ionization class and pKa."""
    mol_copy = deepcopy(mol_no_hs)
    for at in mol_copy.GetAtoms():
        idx = at.GetIdx()
        if idx in acid_dict:
            at.SetProp("ionization", "A")
            at.SetProp("pKa", str(acid_dict[idx]))
        elif idx in base_dict:
            at.SetProp("ionization", "B")
            at.SetProp("pKa", str(base_dict[idx]))
        else:
            at.SetProp("ionization", "O")
    return mol_copy


def get_pKa_data(mol, ph, tph):
    stable_data, unstable_data = [], []
    for at in mol.GetAtoms():
        props = at.GetPropsAsDict()
        acid_or_basic = props.get('ionization', False)
        pKa = float(props.get('pKa', False)) if props.get('pKa', False) else 0.0
        idx = at.GetIdx()

        if acid_or_basic == "A":
            if pKa < ph - tph:
                stable_data.append([idx, pKa, "A"])
            elif ph - tph <= pKa <= ph + tph:
                unstable_data.append([idx, pKa, "A"])
        elif acid_or_basic == "B":
            if pKa > ph + tph:
                stable_data.append([idx, pKa, "B"])
            elif ph - tph <= pKa <= ph + tph:
                unstable_data.append([idx, pKa, "B"])
    return stable_data, unstable_data


def modify_acid(at):
    at.SetFormalCharge(at.GetFormalCharge() - 1)


def modify_base(at):
    at.SetFormalCharge(at.GetFormalCharge() + 1)


def modify_stable_pka(new_mol, stable_data):
    for idx, pka, acid_or_basic in stable_data:
        at = new_mol.GetAtomWithIdx(idx)
        if acid_or_basic == "A":
            modify_acid(at)
        elif acid_or_basic == "B":
            modify_base(at)


def modify_unstable_pka(mol, unstable_data, i):
    combine_pka_datas = list(combinations(unstable_data, i))
    new_unsmis = []
    for pka_datas in combine_pka_datas:
        new_mol = deepcopy(mol)
        if not pka_datas: continue
        for idx, pka, acid_or_basic in pka_datas:
            at = new_mol.GetAtomWithIdx(idx)
            if acid_or_basic == "A":
                modify_acid(at)
            elif acid_or_basic == "B":
                modify_base(at)

        new_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(new_mol)
        new_unsmis.append(Chem.MolToSmiles(new_mol))
    return new_unsmis


def protonate_mol(model, smi, ph, tph):
    """
    Given a SMILES string, a pH, and a threshold (tph), yields SMILES strings for dominant protonation states.
    """
    omol = Chem.MolFromSmiles(smi)
    pred = model.predict(omol, uncharged=True)
    mc = modify_mol(pred["mol"], pred["acid_pka"], pred["base_pka"])

    stable_data, unstable_data = get_pKa_data(mc, ph, tph)
    new_smis = []
    n = len(unstable_data)

    if n == 0:
        new_mol = deepcopy(mc)
        modify_stable_pka(new_mol, stable_data)
        new_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(new_mol)
        new_smis.append(Chem.MolToSmiles(new_mol))
    else:
        for i in range(n + 1):
            new_mol = deepcopy(mc)
            modify_stable_pka(new_mol, stable_data)
            new_smis.extend(modify_unstable_pka(new_mol, unstable_data, i))

    return new_smis

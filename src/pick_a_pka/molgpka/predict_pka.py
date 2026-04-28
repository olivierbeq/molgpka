#!/usr/bin/env python
# coding: utf-8

import math

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize

import torch
import numpy as np
from importlib import resources

from .utils.ionization_group import get_ionization_aid
from .utils.descriptor import mol2vec
from .utils.net import GCNNet
import pick_a_pka.models


def _load_model(model_file, device="cpu"):
    model= GCNNet().to(device)
    # weights_only=True is strictly required in modern PyTorch for security
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    return model

def _model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka

def _predict_acid(mol):
    with resources.as_file(resources.files(pick_a_pka.models).joinpath('weight_acid.pth')) as model_file:
        model_acid = _load_model(model_file)

    acid_idxs= get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = _model_pred(mol, aid, model_acid)
        acid_res.update({aid:apka})
    return acid_res

def _predict_base(mol):
    with resources.as_file(resources.files(pick_a_pka.models).joinpath('weight_base.pth')) as model_file:
        model_base = _load_model(model_file)

    base_idxs= get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = _model_pred(mol, aid, model_base)
        base_res.update({aid:bpka})
    return base_res

def predict(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = _predict_base(mol)
    acid_dict = _predict_acid(mol)
    # Remap without hydrogens
    mol, base_dict, acid_dict = _remap_pka_without_hs(mol, base_dict, acid_dict)
    return base_dict, acid_dict, mol

def _predict_for_protonate(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = _predict_base(mol)
    acid_dict = _predict_acid(mol)
    return base_dict, acid_dict, mol


def _remap_pka_without_hs(mol_with_hs: Chem.Mol, base_pka_dict: dict, acid_pka_dict: dict) -> tuple[Chem.Mol, dict, dict]:
    """
    Remap pKa atom indices in a molecule with explicit hydrogens to the molecule without hydrogens.

    :return: the translated pKa dictionaries and the molecule without Hs.
    """
    # 1. Tag every atom with its original index
    for atom in mol_with_hs.GetAtoms():
        atom.SetIntProp("OrigIdx", atom.GetIdx())

    # 2. Map Hydrogen indices to their Heavy Atom neighbors
    # (Just in case the pKa dictionary points to the H's instead of the O's or N's)
    h_to_heavy = {}
    for atom in mol_with_hs.GetAtoms():
        if atom.GetAtomicNum() == 1:  # If it is a Hydrogen
            neighbors = atom.GetNeighbors()
            if neighbors:
                # Store: H_index -> Heavy_Atom_index
                h_to_heavy[atom.GetIdx()] = neighbors[0].GetIdx()

    # 3. Safely remove Hydrogens
    mol_no_hs = Chem.RemoveHs(mol_with_hs)

    # 4. Create a mapping from Original Index -> New Index for the remaining atoms
    orig_to_new_idx = {}
    for atom in mol_no_hs.GetAtoms():
        if atom.HasProp("OrigIdx"):
            orig_idx = atom.GetIntProp("OrigIdx")
            orig_to_new_idx[orig_idx] = atom.GetIdx()

    # 5. Translate the pKa dictionary
    new_acid_pka_dict = {}
    new_base_pka_dict = {}
    for old_idx, pka_val in acid_pka_dict.items():
        if old_idx in orig_to_new_idx:
            # The pKa was attached to a Heavy Atom that survived
            new_idx = orig_to_new_idx[old_idx]
            new_acid_pka_dict[new_idx] = pka_val

        elif old_idx in h_to_heavy:
            # The pKa was attached to a Hydrogen!
            # Find the heavy atom it belonged to, and use THAT atom's new index
            heavy_orig_idx = h_to_heavy[old_idx]
            if heavy_orig_idx in orig_to_new_idx:
                new_idx = orig_to_new_idx[heavy_orig_idx]
                new_acid_pka_dict[new_idx] = pka_val
    for old_idx, pka_val in base_pka_dict.items():
        if old_idx in orig_to_new_idx:
            # The pKa was attached to a Heavy Atom that survived
            new_idx = orig_to_new_idx[old_idx]
            new_base_pka_dict[new_idx] = pka_val

        elif old_idx in h_to_heavy:
            # The pKa was attached to a Hydrogen!
            # Find the heavy atom it belonged to, and use THAT atom's new index
            heavy_orig_idx = h_to_heavy[old_idx]
            if heavy_orig_idx in orig_to_new_idx:
                new_idx = orig_to_new_idx[heavy_orig_idx]
                new_base_pka_dict[new_idx] = pka_val

    return mol_no_hs, new_base_pka_dict, new_acid_pka_dict


def _generate_microspecies_sequence(mol: Chem.Mol, base_pka_dict: dict, acid_pka_dict: dict) -> list[Chem.Mol]:
    """Generate a list of RDKit molecules representing the sequential deprotonation
    states from pH 0 to pH 14+.
    """
    # Remove explicit Hs to allow safe formal charge modifications
    mol_no_hs = Chem.RemoveHs(mol)

    # Combine and sort all ionizable sites from Lowest pKa to Highest pKa
    ionizable_sites = []
    for idx, pka in base_pka_dict.items():
        ionizable_sites.append((pka, idx, 'base'))
    for idx, pka in acid_pka_dict.items():
        ionizable_sites.append((pka, idx, 'acid'))

    ionizable_sites.sort(key=lambda x: x[0])

    # Pre-calculate the "Fully Protonated" charge for each atom.
    # An atom's highest possible charge is equal to the number of basic pKas it has.
    unique_ionizable_atoms = set([idx for _, idx, _ in ionizable_sites])
    fully_protonated_charges = {}

    for atom_idx in unique_ionizable_atoms:
        # Count how many times this atom acts as a base
        base_count = sum(1 for _, i, site_type in ionizable_sites if i == atom_idx and site_type == 'base')
        fully_protonated_charges[atom_idx] = base_count

    microspecies_list = []

    # Generate State k for k = 0 to N (k = number of lost protons)
    for k in range(len(ionizable_sites) + 1):
        rw_mol = Chem.RWMol(mol_no_hs)

        # Wipe Aromaticity Flags Before Charge Modification
        # It ensures SanitizeMol does not throw a KekulizeException when the pi-electron counts change
        Chem.Kekulize(rw_mol, clearAromaticFlags=True)

        # Step A: Reset everything to FULLY PROTONATED (State 0)
        for atom_idx in unique_ionizable_atoms:
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            # Clear hardcoded H-counts so RDKit calculates them dynamically
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            # Apply the maximum protonated charge (+1 for bases/amphoteric, 0 for pure acids)
            atom.SetFormalCharge(fully_protonated_charges[atom_idx])

        # Step B: Apply Deprotonation for the first 'k' sites
        for i in range(k):
            pka, idx, site_type = ionizable_sites[i]
            atom = rw_mol.GetAtomWithIdx(idx)
            # Every deprotonation event simply subtracts 1 from the formal charge.
            current_charge = atom.GetFormalCharge()
            atom.SetFormalCharge(current_charge - 1)

        # Step C: Recalculate implicit Hydrogens and validate
        rw_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(rw_mol)
        microspecies_list.append(rw_mol.GetMol())

    return microspecies_list


def calculate_microspecies_abundances(mol: Chem.Mol, ph: float = None, ph_range: tuple = None, ph_step: float = None
                                      ) -> dict[float, Chem.Mol] | dict[float, dict[float, Chem.Mol]]:
    """Calculate the relative abundance of each micro-species.

    :param ph: A single pH value to determine the relative abundance of molecular micro-species at.
    :param ph_range: A range of pH to determine the relative abundance of molecular micro-species at. Ignored if `ph` is not None.
    :param ph_step: The incremental step to consider between values of the `ph_range`. Ignored if ph_range is None.
    :return: If `ph` is not None, then a dictionary with relative abundances as keys and micro-species molecules as values {relative_abundance_float: Chem.Mol}.
    If `ph_range` is not None, then a dictionary with incremental pH steps as keys and dictionaries as values,
    each with relative abundances as keys and corresponding micro-species molecules as values {pH_float: {relative_abundance_float: Chem.Mol}}.
    """
    if ph is None and ph_range is None:
        raise ValueError("Either `ph` or `ph_range` must be specified.")
    if ph_range is not None and ph_step is None:
        raise ValueError("`ph_step` must be specified when using `ph_range`.")

    base_pka_dict, acid_pka_dict, mol = predict(mol, uncharged=True)
    # Use the generator to get the ordered state molecules
    microspecies = _generate_microspecies_sequence(mol, base_pka_dict, acid_pka_dict)

    all_pkas = sorted(list(base_pka_dict.values()) + list(acid_pka_dict.values()))
    n_pkas = len(all_pkas)

    def get_abundances_at_ph(current_ph):
        log_ratios = [0.0]  # Fully protonated state
        current_sum_pka = 0.0

        # Calculate P_k / P_0 log ratios
        for k in range(1, n_pkas + 1):
            current_sum_pka += all_pkas[k - 1]
            log_ratio = k * current_ph - current_sum_pka
            log_ratios.append(log_ratio)

        # Prevent float overflow
        max_log = max(log_ratios)
        ratios = [10 ** (lr - max_log) for lr in log_ratios]

        # Normalize
        total_ratio = sum(ratios)
        fractions = [(r / total_ratio) * 100.0 for r in ratios]

        # Format output: {Abundance: Molecule}
        result = {}
        for frac, state_mol in zip(fractions, microspecies):
            key = frac
            # Infinitesimal offset ensures no key clashing if two states have identical abundance (e.g. 0.0)
            while key in result:
                key += 1e-12
            result[key] = state_mol
        return result

    # Evaluate execution mode based on provided arguments
    if ph is not None:
        return get_abundances_at_ph(ph)

    else:
        results_over_range = {}
        # np.arange used to avoid float precision issues during iteration
        for current_ph in np.arange(ph_range[0], ph_range[1] + (ph_step / 2), ph_step):
            # Round pH key to avoid ugly dict keys like 0.100000000001
            rounded_ph = round(current_ph, max(0, int(math.ceil(-math.log10(ph_step)))))
            results_over_range[rounded_ph] = get_abundances_at_ph(current_ph)

        return results_over_range

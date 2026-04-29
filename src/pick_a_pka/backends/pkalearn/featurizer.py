import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
import copy

from .utils import swap_bond_atoms

electronegativity = {'C': 6.27, 'N': 7.30, 'O': 7.54, 'F': 10.41, 'H': 7.18, 'Cl': 8.30, 'S': 6.22, 'Br': 7.59,
                     'I': 6.76, 'P': 5.62, 'B': 4.29, 'Si': 4.77, 'Se': 5.89, 'As': 5.30, 'Max': 10.41, 'Min': 4.29,
                     'Range': 6.12}
hardness = {'C': 5.00, 'N': 7.23, 'O': 6.08, 'F': 7.01, 'H': 6.43, 'S': 4.14, 'Cl': 4.68, 'Br': 4.22, 'I': 3.69,
            'P': 4.88, 'B': 4.01, 'Si': 3.38, 'Se': 3.87, 'As': 4.50, 'Max': 7.23, 'Min': 3.38, 'Range': 3.85}
atom_diameter = {'C': 75.0, 'N': 71.0, 'O': 63.0, 'F': 64.0, 'H': 32.0, 'S': 103.0, 'Cl': 99.0, 'Br': 114.0, 'I': 133.0,
                 'P': 111.0, 'B': 85.0, 'Si': 116.0, 'Se': 116.0, 'As': 121.0, 'Max': 133.0, 'Min': 32.0,
                 'Range': 101.0}


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_node_features(mol, center, config):
    ring = mol.GetRingInfo()
    mol_nodes_features = []

    for atom in mol.GetAtoms():
        node_features = []
        if config['atom_feature_element']:
            node_features += one_hot(atom.GetSymbol(), ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'])

        if config['atom_feature_electronegativity']:
            en = (electronegativity.get(atom.GetSymbol(), electronegativity['Min']) - electronegativity['Min']) / \
                 electronegativity['Range']
            node_features += [en]

        if config['atom_feature_hardness']:
            hn = (hardness.get(atom.GetSymbol(), hardness['Min']) - hardness['Min']) / hardness['Range']
            node_features += [hn]

        if config['atom_feature_atom_size']:
            ad = (atom_diameter.get(atom.GetSymbol(), atom_diameter['Min']) - atom_diameter['Min']) / atom_diameter[
                'Range']
            node_features += [ad]

        if config['atom_feature_hybridization']:
            node_features += one_hot(atom.GetHybridization(),
                                     [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                      Chem.rdchem.HybridizationType.SP3]
                                     )

        if config['atom_feature_aromaticity']:
            node_features += [atom.GetIsAromatic()]

        if config['atom_feature_number_of_rings']:
            node_features += one_hot(ring.NumAtomRings(atom.GetIdx()), [0, 1, 2])

        if config['atom_feature_ring_size']:
            node_features += [ring.IsAtomInRingOfSize(atom.GetIdx(), i) for i in [3, 4, 5]]
            node_features += [any(ring.IsAtomInRingOfSize(atom.GetIdx(), i) for i in [6, 7, 8, 9, 10])]

        if config['atom_feature_number_of_Hs']:
            node_features += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3])

        if config['atom_feature_formal_charge']:
            node_features += one_hot(atom.GetFormalCharge(), [-1, 0, 1])

        node_features += [1 if int(atom.GetIdx()) == center else 0]
        mol_nodes_features.append(node_features)

    return torch.tensor(np.array(mol_nodes_features), dtype=torch.float)


def get_edge_features(mol, config):
    edges_features = []
    for bond in mol.GetBonds():
        edge_f1, edge_f2 = [], []
        if config['bond_feature_bond_order']:
            edge_f1 += one_hot(bond.GetBondTypeAsDouble(), [1, 1.5, 2, 3])
            edge_f2 += one_hot(bond.GetBondTypeAsDouble(), [1, 1.5, 2, 3])

        if config['bond_feature_conjugation'] and not config['bond_feature_charge_conjugation'] and not config[
            'bond_feature_focused']:
            edge_f1.append(bond.GetIsConjugated())
            edge_f2.append(bond.GetIsConjugated())

        if config['bond_feature_polarization']:
            el1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
            el2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
            pol = (electronegativity.get(el1, 0) - electronegativity.get(el2, 0)) / electronegativity[
                'Range'] if el1 in electronegativity and el2 in electronegativity else 0
            edge_f1 += [pol]
            edge_f2 += [-pol]

        if config['bond_feature_charge_conjugation'] or config['bond_feature_focused']:
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            central_atom = -1
            peripheral_atom = -1
            strongConjugation = 0
            weakConjugation = 0

            b_type = bond.GetBondTypeAsDouble()
            if b_type == 1 or b_type == 2:
                sym1 = mol.GetAtomWithIdx(atom1).GetSymbol()
                sym2 = mol.GetAtomWithIdx(atom2).GetSymbol()
                if sym1 in ["O", "N"]:
                    central_atom, peripheral_atom = atom2, atom1
                elif sym2 in ["O", "N"]:
                    central_atom, peripheral_atom = atom1, atom2

                if central_atom != -1 and mol.GetAtomWithIdx(central_atom).GetSymbol() == "C":
                    p_charge = mol.GetAtomWithIdx(peripheral_atom).GetFormalCharge()

                    if b_type == 1:
                        if p_charge == -1:
                            for b2 in mol.GetBonds():
                                if b2.GetBondTypeAsDouble() == 1: continue
                                a1, a2 = b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()
                                if (a1 == central_atom and mol.GetAtomWithIdx(a2).GetSymbol() in ["O", "N"]) or \
                                        (a2 == central_atom and mol.GetAtomWithIdx(a1).GetSymbol() in ["O", "N"]):
                                    strongConjugation = 1
                                    break
                        elif p_charge == 0:
                            for b2 in mol.GetBonds():
                                if b2.GetBondTypeAsDouble() == 1: continue
                                a1, a2 = b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()
                                if (a1 == central_atom and mol.GetAtomWithIdx(a2).GetFormalCharge() == 1) or \
                                        (a2 == central_atom and mol.GetAtomWithIdx(a1).GetFormalCharge() == 1):
                                    strongConjugation = 1
                                    break
                                if (a1 == central_atom and mol.GetAtomWithIdx(a2).GetFormalCharge() == 0) or \
                                        (a2 == central_atom and mol.GetAtomWithIdx(a1).GetFormalCharge() == 0):
                                    weakConjugation = 1
                                    break
                    elif b_type == 2:
                        if p_charge == 1:
                            for b2 in mol.GetBonds():
                                if b2.GetBondTypeAsDouble() != 1: continue
                                a1, a2 = b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()
                                if (a1 == central_atom and mol.GetAtomWithIdx(a2).GetSymbol() in ["O", "N"]) or \
                                        (a2 == central_atom and mol.GetAtomWithIdx(a1).GetSymbol() in ["O", "N"]):
                                    strongConjugation = 1
                                    break
                        elif p_charge == 0:
                            for b2 in mol.GetBonds():
                                if b2.GetBondTypeAsDouble() != 1: continue
                                a1, a2 = b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()
                                if (a1 == central_atom and mol.GetAtomWithIdx(a2).GetFormalCharge() == -1) or \
                                        (a2 == central_atom and mol.GetAtomWithIdx(a1).GetFormalCharge() == -1):
                                    strongConjugation = 1
                                    break
                                if (a1 == central_atom and mol.GetAtomWithIdx(a2
                                                                              ).GetFormalCharge() == 0 and mol.GetAtomWithIdx(
                                        a2
                                        ).GetSymbol() in ["O", "N"]) or \
                                        (a2 == central_atom and mol.GetAtomWithIdx(a1
                                                                                   ).GetFormalCharge() == 0 and mol.GetAtomWithIdx(
                                            a1
                                            ).GetSymbol() in ["O", "N"]):
                                    weakConjugation = 1

            if config['bond_feature_conjugation']:
                if not config['bond_feature_charge_conjugation'] and strongConjugation == 1:
                    weakConjugation = 1
                edge_f1 += [weakConjugation]
                edge_f2 += [weakConjugation]

            if config['bond_feature_charge_conjugation']:
                edge_f1 += [strongConjugation]
                edge_f2 += [strongConjugation]

        edges_features += [edge_f1, edge_f2]

    return torch.tensor(np.array(edges_features), dtype=torch.float)


def get_edge_info(mol_A, mol_B, center, distance_matrix):
    edge_indices = []
    mol_A, mol_B = swap_bond_atoms(mol_A, mol_B, center, distance_matrix)
    for bond in mol_A.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
    edge_indices = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    return edge_indices, mol_A, mol_B


def from_acid_to_base(mol, center):
    base_found = False
    for atom in mol.GetAtoms():
        if int(atom.GetIdx()) == center:
            if atom.GetTotalNumHs() > 0 or atom.GetSymbol() == 'C':
                atom.SetFormalCharge(atom.GetFormalCharge() - 1)
                base_found = True
                if atom.GetNumExplicitHs() > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
            break
    smile_base = "none"
    if base_found:
        Chem.SanitizeMol(mol, catchErrors=True)
        smile_base = Chem.MolToSmiles(mol)
    return base_found, mol, smile_base


def mol_to_graph(mol_obj_A, center, config):
    """Generates the torch_geometric Data object for a single molecule and center."""
    mol_obj_B = copy.deepcopy(mol_obj_A)
    base_found, mol_obj_B, smiles_B = from_acid_to_base(mol_obj_B, center)
    if not base_found:
        return None

    node_features_A = get_node_features(mol_obj_A, center, config)
    node_features_B = get_node_features(mol_obj_B, center, config)

    distance_matrix = rdmolops.GetDistanceMatrix(mol_obj_A)
    edge_index, mol_obj_A, mol_obj_B = get_edge_info(mol_obj_A, mol_obj_B, center, distance_matrix)

    edge_features_A = get_edge_features(mol_obj_A, config)
    edge_features_B = get_edge_features(mol_obj_B, config)

    # Base checkpoint uses acid_or_base='base' logically, meaning we slice B features
    if config['acid_or_base'] == "acid":
        edge_features = edge_features_A
        node_features = node_features_A
    elif config['acid_or_base'] == "base":
        edge_features = edge_features_B
        node_features = node_features_B
    else:
        edge_features = torch.cat([edge_features_A, edge_features_B], axis=1)
        node_features = node_features_A

    mol_formal_charge = torch.tensor(rdmolops.GetFormalCharge(mol_obj_A), dtype=torch.float)
    center_formal_charge = torch.tensor(mol_obj_A.GetAtomWithIdx(center).GetFormalCharge(), dtype=torch.float)

    local_atoms = np.where(distance_matrix[center] <= config['mask_size'])[0]
    node_index = torch.tensor(local_atoms, dtype=torch.long)

    # Move center to 0 to align attention mask
    if center != 0:
        node_features[[center, 0]] = node_features[[0, center]]
        edge_index[edge_index == center] = 999
        edge_index[edge_index == 0] = center
        edge_index[edge_index == 999] = 0

    return Data(x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                node_index=node_index,
                mol_formal_charge=mol_formal_charge,
                center_formal_charge=center_formal_charge,
                batch=torch.zeros(len(node_features), dtype=torch.long)
                )

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from importlib import resources


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def get_atom_features(mol, aid):
    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)

    acceptor_smarts_one = '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
    acceptor_smarts_two = "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    donor_smarts_one = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
    donor_smarts_two = "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]"

    hydrogen_donor_one = Chem.MolFromSmarts(donor_smarts_one)
    hydrogen_donor_two = Chem.MolFromSmarts(donor_smarts_two)
    hydrogen_acceptor_one = Chem.MolFromSmarts(acceptor_smarts_one)
    hydrogen_acceptor_two = Chem.MolFromSmarts(acceptor_smarts_two)

    hydrogen_donor_match = list(set(mol.GetSubstructMatches(hydrogen_donor_one) +
                                    mol.GetSubstructMatches(hydrogen_donor_two)
                                    )
                                )
    hydrogen_acceptor_match = list(set(mol.GetSubstructMatches(hydrogen_acceptor_one) +
                                       mol.GetSubstructMatches(hydrogen_acceptor_two)
                                       )
                                   )

    ring = mol.GetRingInfo()

    m = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        o = []
        o += one_hot(atom.GetSymbol(), ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I'])
        o += [atom.GetDegree()]
        o += one_hot(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]
                     )
        o += [atom.GetValence(Chem.ValenceType.IMPLICIT)]
        o += [atom.GetIsAromatic()]
        o += [ring.IsAtomInRingOfSize(atom_idx, 3), ring.IsAtomInRingOfSize(atom_idx, 4),
              ring.IsAtomInRingOfSize(atom_idx, 5), ring.IsAtomInRingOfSize(atom_idx, 6),
              ring.IsAtomInRingOfSize(atom_idx, 7), ring.IsAtomInRingOfSize(atom_idx, 8)]

        o += [atom_idx in hydrogen_donor_match]
        o += [atom_idx in hydrogen_acceptor_match]
        o += [atom.GetFormalCharge()]
        o += [0 if atom_idx == aid else len(Chem.rdmolops.GetShortestPath(mol, atom_idx, aid))]
        o += [atom_idx == aid]
        m.append(o)
    return m


def mol_to_graph(mol, atom_idx, pka=None):
    """Convert RDKit Mol to torch_geometric graph."""
    node_f = get_atom_features(mol, atom_idx)
    edge_index = get_bond_pair(mol)
    batch = np.zeros(len(node_f), )

    data = Data(x=torch.tensor(node_f, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                batch=torch.tensor(batch, dtype=torch.long)
                )

    if pka is not None:
        data.pka = torch.tensor([[pka]], dtype=torch.float)
    return data


def split_acid_base_pattern():
    pkg = resources.files("pick_a_pka.backends.molgpka")
    with resources.as_file(pkg.joinpath("smarts_pattern.tsv")) as smarts_file:
        df_smarts = pd.read_csv(smarts_file, sep="\t")
    df_smarts_acid = df_smarts[df_smarts.Acid_or_base == "A"]
    df_smarts_base = df_smarts[df_smarts.Acid_or_base == "B"]
    return df_smarts_acid, df_smarts_base


def unique_acid_match(matches):
    single_matches = list(set([m[0] for m in matches if len(m) == 1]))
    double_matches = [m for m in matches if len(m) == 2]
    single_matches = [[j] for j in single_matches]
    double_matches.extend(single_matches)
    return double_matches


def match_acid(df_smarts_acid, mol):
    matches = []
    for idx, name, smarts, index, acid_base in df_smarts_acid.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if not match: continue

        if isinstance(index, str) and ',' in index:
            index_list = [int(i) for i in index.split(",")]
            for m in match: matches.append([m[index_list[0]], m[index_list[1]]])
        else:
            idx_int = int(index)
            for m in match: matches.append([m[idx_int]])

    matches = unique_acid_match(matches)
    return [j for i in matches for j in i]


def match_base(df_smarts_base, mol):
    matches = []
    for idx, name, smarts, indexs, acid_base in df_smarts_base.itertuples():
        pattern = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if not match: continue

        for index in str(indexs).split(","):
            idx_int = int(index)
            for m in match: matches.append([m[idx_int]])

    matches = unique_acid_match(matches)
    return [j for i in matches for j in i]


def get_ionization_aid(mol, acid_or_base=None):
    if mol is None:
        raise RuntimeError("Invalid RDKit molecule")

    df_smarts_acid, df_smarts_base = split_acid_base_pattern()
    acid_matches = match_acid(df_smarts_acid, mol)
    base_matches = match_base(df_smarts_base, mol)

    if acid_or_base == "acid":
        return acid_matches
    elif acid_or_base == "base":
        return base_matches
    return acid_matches, base_matches

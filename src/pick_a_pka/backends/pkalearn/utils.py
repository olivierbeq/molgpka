def swap_tensor_items(original_tensor, from_item_idx, to_item_idx):
    new_tensor = original_tensor.clone()
    new_tensor[[from_item_idx, to_item_idx]] = original_tensor[[to_item_idx, from_item_idx]]
    return new_tensor


def swap_tensor_columns(original_tensor, from_idx, to_idx, tensor_size):
    new_tensor = original_tensor.clone()
    new_tensor[:, [from_idx, to_idx]] = original_tensor[:, [to_idx, from_idx]]
    return new_tensor


def swap_tensor_values(original_tensor, from_idx, to_idx):
    new_tensor = original_tensor.clone()
    indexes_old = (original_tensor == from_idx)
    new_tensor[indexes_old] = 999

    indexes_old = (original_tensor == to_idx)
    new_tensor[indexes_old] = from_idx

    indexes_old = (original_tensor == 999)
    new_tensor[indexes_old] = to_idx

    return new_tensor


def swap_bond_atoms(mol_A, mol_B, center, distance_matrix):
    from rdkit import Chem
    mw_A = Chem.RWMol(mol_A)
    mw_B = Chem.RWMol(mol_B)

    for i in range(mol_A.GetNumBonds()):
        bond_A = mol_A.GetBondWithIdx(i)
        bond_B = mol_B.GetBondWithIdx(i)
        atom1 = bond_A.GetBeginAtomIdx()
        atom2 = bond_A.GetEndAtomIdx()

        if distance_matrix[atom1][center] > distance_matrix[atom2][center]:
            mw_A.RemoveBond(atom1, atom2)
            mw_B.RemoveBond(atom1, atom2)
            mw_A.AddBond(atom2, atom1, bond_A.GetBondType())
            mw_B.AddBond(atom2, atom1, bond_B.GetBondType())

    return mw_A, mw_B

def whichElement(smiles, j):
    if smiles[j] == 'F' or smiles[j] == 'I' or smiles[j] == 'P':
        return j, smiles[j], 'none', 'none'

    char = 'none'
    char2 = 'none'
    char3 = 'none'
    char4 = 'none'
    charge = 'none'
    brackets = False
    if j < len(smiles) - 1:
        char = smiles[j + 1]
    if j < len(smiles) - 2:
        char2 = smiles[j + 2]
    if j < len(smiles) - 3:
        char3 = smiles[j + 3]
    if j < len(smiles) - 4:
        char4 = smiles[j + 4]
    if j > 0 and smiles[j - 1] == '[':
        brackets = True

    if smiles[j] == 'N' or smiles[j] == 'O' or smiles[j] == 'n' or smiles[j] == 'o' or \
            (smiles[j] == 'C' and char != 'l'):
        element = smiles[j]
        if char == 'H':
            if char2 == '2' or char2 == '3':
                j += 2
                if char3 == '+' or char3 == '-':
                    charge = char3
                    j += 1
                return j, element + char + char2, charge, brackets
            else:
                j += 1
                if char2 == '+' or char2 == '-':
                    charge = char2
                    j += 1
                return j, element + char, charge, brackets
        elif char == '@':
            if char2 == 'H':
                j += 2
                if char3 == '+' or char3 == '-':
                    charge = char3
                    j += 1
                return j, element + char + char2, charge, brackets
            if char2 == ']':
                j += 2
                return j, element + char, charge, brackets
            elif char2 == '@':
                if char3 == 'H':
                    j += 3
                    # print("utils 460", element, char, char2, char3)
                    if char4 == '+' or char4 == '-':
                        charge = char4
                        j += 1
                    return j, element + char + char2 + char3, charge, brackets
                if char3 == ']':
                    j += 3
                    return j, element + char + char2, charge, brackets
            else:
                if char2 == '+' or char2 == '-':
                    charge = char2
                    j += 1
                return j, element + char, charge, brackets
        elif char == '+' or char == '-':
            charge = char
            j += 1
            return j, element, charge, brackets
        else:
            return j, smiles[j], charge, brackets

    if smiles[j] == 'c':
        return j, 'C', charge, brackets

    if smiles[j] == 'C' and char == 'l':
        if char2 == '-':
            charge = '-'
            j += 1
        j += 1
        return j, 'Cl', charge, brackets

    if smiles[j] == 'S' or smiles[j] == 's':
        if char == 'e':
            j += 1
            return j, 'Se', charge, brackets
        elif char == 'i':
            j += 1
            return j, 'Si', charge, brackets
        else:
            if char == '-':
                charge = '-'
                j += 1
            return j, 'S', charge, brackets

    if smiles[j] == 'B':
        if char == 'r':
            if char2 == '-':
                charge = '-'
                j += 1
            j += 1
            return j, 'Br', charge, brackets
        else:
            return j, 'B', charge, brackets

    if smiles[j] == 'A':
        if char == 's':
            j += 1
            return j, 'As', charge, brackets
        else:
            return j, 'Al', charge, brackets
    return j, 'none', charge, brackets

import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D


# Old Function
def orient_canonically(mol):
    # Standardize
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol, sampleSeed=0, nSample=100, forceRDKit=True)
    conf = mol.GetConformer()
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    # PCA Alignment to Horizontal
    pos = conf.GetPositions()[:, :2]
    pos -= np.mean(pos, axis=0)
    evals, evecs = np.linalg.eigh(np.cov(pos, rowvar=False))

    # Handle the "Short Axis" fallback for round molecules
    if evals[0] / evals[1] > 0.9:
        rank_0_pos = pos[ranks.index(0)]
        angle = np.arctan2(rank_0_pos[1], rank_0_pos[0])
        rot = -angle  # Points Rank 0 to the right initially
    else:
        rot = -np.arctan2(evecs[1, 1], evecs[0, 1])

    # Apply rotation
    c, s = np.cos(rot), np.sin(rot)
    pos = np.dot(pos, np.array([[c, -s], [s, c]]).T)

    # Decisive Symmetry Breaking (Flipping)
    # Calculate weighted moments to determine "Up/Down" and "Left/Right"
    # Use exponential weights so high-rank atoms dominate the decision
    weights = np.power(np.array(ranks) + 1, 2)
    m_x = np.sum(pos[:, 0] * weights)
    m_y = np.sum(pos[:, 1] * weights)

    # Goal: High-rank atoms on the Left (m_x < 0)
    if m_x > 0:
        pos[:, 0] *= -1
    # Goal: High-rank atoms on the Top (m_y > 0)
    if m_y < 0:
        pos[:, 1] *= -1

    # Update conformer
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (pos[i, 0], pos[i, 1], 0))

    return mol


def get_score(conf):
    """Minimize the longest dimension to pack into a square."""
    pos = conf.GetPositions()
    min_c = np.min(pos, axis=0)
    max_c = np.max(pos, axis=0)
    return max(max_c[0] - min_c[0], max_c[1] - min_c[1])


def ccw(A, B, C):
    """Helper for line intersection."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """Returns True if line segment AB intersects segment CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def point_segment_distance(p, a, b):
    """
    Calculates the shortest distance from point p to the line segment a-b.
    All arguments are numpy arrays.
    """
    v = b - a
    w = p - a

    # Project w onto v
    c1 = np.dot(w, v)
    if c1 <= 0:
        # Closest point on the segment is 'a'
        return np.linalg.norm(p - a)

    c2 = np.dot(v, v)
    if c2 <= c1:
        # Closest point on the segment is 'b'
        return np.linalg.norm(p - b)

    # Closest point is strictly between 'a' and 'b'
    b_ratio = c1 / c2
    proj = a + b_ratio * v
    return np.linalg.norm(p - proj)


def has_crossings_or_collisions(mol, conf, atom_bond_clearance=0.85):
    """Checks for overlapping atoms, crossing bonds, AND atom-bond collisions."""
    pos = conf.GetPositions()
    n_atoms = mol.GetNumAtoms()
    bonds = mol.GetBonds()
    n_bonds = len(bonds)

    # 1. Check Atom-Atom Collisions (Threshold 0.5 Å)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                return True

    # 2. Check Bond-Bond Intersections
    for i in range(n_bonds):
        for j in range(i + 1, n_bonds):
            b1, b2 = bonds[i], bonds[j]

            # Skip if bonds share an atom
            if len(set((b1.GetBeginAtomIdx(), b1.GetEndAtomIdx(),
                        b2.GetBeginAtomIdx(), b2.GetEndAtomIdx())
                       )
                   ) < 4:
                continue

            p1 = pos[b1.GetBeginAtomIdx()]
            p2 = pos[b1.GetEndAtomIdx()]
            p3 = pos[b2.GetBeginAtomIdx()]
            p4 = pos[b2.GetEndAtomIdx()]

            if intersect(p1, p2, p3, p4):
                return True

    # 3. NEW: Check Atom-Bond proximity (Text Label overlaps)
    for i in range(n_atoms):
        p_atom = pos[i]

        # We enforce a slightly larger clearance if the atom is a Heteroatom
        # or terminal, as RDKit will definitely draw a text label for these.
        atom_obj = mol.GetAtomWithIdx(i)
        if atom_obj.GetAtomicNum() != 6 or atom_obj.GetDegree() == 1:
            clearance = atom_bond_clearance * 1.2  # Make extra room for "OH", "NH2", etc.
        else:
            clearance = atom_bond_clearance

        for b in bonds:
            # Skip if this atom is part of the bond
            if b.GetBeginAtomIdx() == i or b.GetEndAtomIdx() == i:
                continue

            p_a = pos[b.GetBeginAtomIdx()]
            p_b = pos[b.GetEndAtomIdx()]

            # If the bond passes too close to the atom, reject it
            if point_segment_distance(p_atom, p_a, p_b) < clearance:
                return True

    return False


def get_branch_atoms(mol, pivot_u, start_v):
    """Finds all atoms on the 'v' side of the u-v bond."""
    stack = [start_v]
    visited = {pivot_u, start_v}
    branch_atoms = []
    while stack:
        curr = stack.pop()
        branch_atoms.append(curr)
        for neighbor in mol.GetAtomWithIdx(curr).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx not in visited:
                visited.add(idx)
                stack.append(idx)
    return branch_atoms


def reflect_branch_2d(conf, u_idx, v_idx, branch_atoms):
    """Reflects a branch across the axis defined by bond u-v.       This perfectly preserves standard 2D bond angles (120/180 deg)."""
    p1 = np.array(conf.GetAtomPosition(u_idx))
    p2 = np.array(conf.GetAtomPosition(v_idx))
    # Vector of the bond
    L = p2 - p1
    L_norm = np.linalg.norm(L)
    if L_norm < 1e-4: return
    u_vec = L / L_norm
    for idx in branch_atoms:
        # Don't move the atoms that are exactly ON the reflection axis
        if idx == u_idx or idx == v_idx:
            continue
        P = np.array(conf.GetAtomPosition(idx))
        V = P - p1
        # Calculate reflection
        proj_length = np.dot(V, u_vec)
        proj = proj_length * u_vec
        V_perp = V - proj
        # Mirror the perpendicular component
        P_new = P - 2 * V_perp
        conf.SetAtomPosition(idx, Point3D(*P_new))


def canonicalize_orientation(mol):
    """
    Evaluates the 8 possible 2D orientations (rotations + flips)    and chooses a single deterministic layout based on canonical atom ranks.
    """
    # 1. Straighten the molecule to align tightly with X/Y axes
    rdDepictor.StraightenDepiction(mol)
    conf = mol.GetConformer()
    # 2. Get intrinsic canonical ranks for the atoms
    ranks = list(Chem.CanonicalRankAtoms(mol))
    # Center the molecule's bounding box at (0,0)
    pos = conf.GetPositions()
    min_c = np.min(pos, axis=0)
    max_c = np.max(pos, axis=0)
    center = (min_c + max_c) / 2.0
    for i in range(mol.GetNumAtoms()):
        p = pos[i]
        conf.SetAtomPosition(i, Point3D(p[0] - center[0], p[1] - center[1], 0))
    original_pos = conf.GetPositions()
    # The 8 possible transformations (D4 symmetry group)
    # 4 Rotations + 4 Flips
    transforms = [
        lambda x, y: (x, y),  # Original
        lambda x, y: (-y, x),  # 90 deg
        lambda x, y: (-x, -y),  # 180 deg
        lambda x, y: (y, -x),  # 270 deg
        lambda x, y: (-x, y),  # Flip X (Horizontal flip)
        lambda x, y: (x, -y),  # Flip Y (Vertical flip)
        lambda x, y: (y, x),  # Flip Diagonal 1
        lambda x, y: (-y, -x)  # Flip Diagonal 2
    ]
    best_score = None
    best_pos = None
    for t in transforms:
        # Apply transformation
        new_pos = np.array([[t(p[0], p[1])[0], t(p[0], p[1])[1], 0.0] for p in original_pos])
        # Calculate Width and Height
        min_p = np.min(new_pos, axis=0)
        max_p = np.max(new_pos, axis=0)
        w = max_p[0] - min_p[0]
        h = max_p[1] - min_p[1]
        # CRITERIA 1: Prefer Horizontal (Width >= Height)
        # We give a massive score bonus (1) to horizontal layouts.
        is_horizontal = 1 if (w >= h - 1e-3) else 0
        # CRITERIA 2 & 3: Push highest-ranked atoms to the Right (+X) and Top (+Y)
        # We multiply the atom's rank by its X and Y coordinates.
        score_x = sum(ranks[i] * new_pos[i][0] for i in range(len(ranks)))
        score_y = sum(ranks[i] * new_pos[i][1] for i in range(len(ranks)))
        # CRITERIA 4: Secondary moments for tie-breaking highly symmetric structures
        score_x3 = sum(ranks[i] * (new_pos[i][0] ** 3) for i in range(len(ranks)))
        score_y3 = sum(ranks[i] * (new_pos[i][1] ** 3) for i in range(len(ranks)))
        # Create a tuple score. Python automatically compares tuples left-to-right!
        score = (
            is_horizontal, round(score_x, 2), round(score_y, 2), round(score_x3, 2), round(score_y3, 2)
        )
        if best_score is None or score > best_score:
            best_score = score
            best_pos = new_pos
    # Apply the absolute winning coordinates
    for i, p in enumerate(best_pos):
        conf.SetAtomPosition(i, Point3D(*p))
    # CRITICAL: Because we may have flipped the molecule horizontally/vertically,    # we MUST recalculate the 2D stereocenter wedges to preserve chemical accuracy.
    Chem.WedgeMolBonds(mol, conf)
    return mol


def optimize_sensical_folding(mol, steps=1000):
    # Generate the standard, chemically accurate 2D coordinates
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    # Find rotatable single bonds (not in rings, not terminal)
    rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    rotatable_bonds = mol.GetSubstructMatches(rot_smarts)
    if not rotatable_bonds: return mol
    best_conf_positions = conf.GetPositions()
    current_score = get_score(conf)
    best_score = current_score
    # Simulated Annealing params
    temp = 2.0
    cooling_rate = 0.99
    for _ in range(steps):
        # Pick random bond and randomize which side is the "pivot" vs "branch"
        u, v = random.choice(rotatable_bonds)
        if random.random() > 0.5: u, v = v, u
        branch = get_branch_atoms(mol, u, v)
        prev_positions = conf.GetPositions()
        # Flips the branch across the bond axis
        reflect_branch_2d(conf, u, v, branch)
        new_score = get_score(conf)
        delta = new_score - current_score
        accept = False
        # Make sure no bonds cross and no atoms overlap
        if not has_crossings_or_collisions(mol, conf):
            if delta < 0:
                accept = True
            elif temp > 0 and random.random() < np.exp(-delta / temp):
                accept = True
        if accept:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_conf_positions = conf.GetPositions()
        else:
            # Revert to previous
            for idx, pos in enumerate(prev_positions):
                conf.SetAtomPosition(idx, Point3D(*pos))
        temp *= cooling_rate
    # Apply absolute best found
    for idx, pos in enumerate(best_conf_positions):
        conf.SetAtomPosition(idx, Point3D(*pos))
    rdDepictor.StraightenDepiction(mol)
    mol = canonicalize_orientation(mol)
    return mol

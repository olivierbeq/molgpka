import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor


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

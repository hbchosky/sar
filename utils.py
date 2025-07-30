# File: activity_cliff_app/utils.py
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdFMCS import FindMCS
import numpy as np

def compute_pIC50(ic50_nM):
    return -np.log10(ic50_nM * 1e-9)

def smiles_to_image(smiles, size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=size)

def smiles_diff_to_images(smiles1, smiles2, size=(300, 300)):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None, None # Handle invalid SMILES

    # Find Maximum Common Substructure (MCS)
    mcs_result = FindMCS([mol1, mol2])
    mcs_smarts = mcs_result.smartsString
    
    highlight_atoms1 = []
    highlight_atoms2 = []

    if not mcs_smarts:
        # If no MCS found, highlight all atoms
        highlight_atoms1 = list(range(mol1.GetNumAtoms()))
        highlight_atoms2 = list(range(mol2.GetNumAtoms()))
    else:
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        
        # Get atom indices in mol1 that are part of MCS
        mol1_mcs_matches = mol1.GetSubstructMatches(mcs_mol)
        mol1_mcs_atoms_indices = set()
        for match in mol1_mcs_matches:
            mol1_mcs_atoms_indices.update(match)

        # Get atom indices in mol2 that are part of MCS
        mol2_mcs_matches = mol2.GetSubstructMatches(mcs_mol)
        mol2_mcs_atoms_indices = set()
        for match in mol2_mcs_matches:
            mol2_mcs_atoms_indices.update(match)

        # Identify differing atoms (not part of MCS)
        highlight_atoms1 = [atom.GetIdx() for atom in mol1.GetAtoms() if atom.GetIdx() not in mol1_mcs_atoms_indices]
        highlight_atoms2 = [atom.GetIdx() for atom in mol2.GetAtoms() if atom.GetIdx() not in mol2_mcs_atoms_indices]

    # Generate images with highlighting
    img1 = Draw.MolToImage(mol1, size=size, highlightAtoms=highlight_atoms1, highlightColor=(1, 0, 0))
    img2 = Draw.MolToImage(mol2, size=size, highlightAtoms=highlight_atoms2, highlightColor=(1, 0, 0))

    return img1, img2
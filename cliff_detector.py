# File: activity_cliff_app/cliff_detector.py
# RDKit을 기반으로 분자 유사도(Tanimoto)와 생물학적 활성(IC50 또는 pIC50 등)의 차이를 이용하여 Activity Cliff 후보 쌍을 자동으로 탐지
import streamlit as st # Import streamlit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from itertools import combinations
import pandas as pd

# Correct import for GetMorganGenerator
from rdkit.Chem import rdFingerprintGenerator

# Initialize the MorganGenerator once
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def fingerprint(mol):
    # Use the MorganGenerator to get the fingerprint
    return morgan_gen.GetFingerprint(mol)

def similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

@st.cache_data # Add the cache decorator
def detect_activity_cliffs(df, sim_thres=0.85, act_thres=1.0):
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['fp'] = df['mol'].apply(fingerprint)

    results = []
    for i, j in combinations(df.index, 2):
        sim = similarity(df.loc[i, 'fp'], df.loc[j, 'fp'])
        if sim >= sim_thres:
            diff = abs(df.loc[i, 'pIC50'] - df.loc[j, 'pIC50'])
            if diff >= act_thres:
                results.append({
                    'mol1_idx': i,
                    'mol2_idx': j,
                    'mol1_smiles': df.loc[i, 'SMILES'],
                    'mol2_smiles': df.loc[j, 'SMILES'],
                    'mol1_activity': df.loc[i, 'pIC50'],
                    'mol2_activity': df.loc[j, 'pIC50'],
                    'sim': sim,
                    'activity_diff': diff,
                })
    return pd.DataFrame(results)
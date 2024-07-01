import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

MORGAN_RADIUS = 3
MORGAN_NUM_BITS = 1024

def morgan_binary_features_generator(mol,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:

    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


load_path='./dataset/SMILES/'
save_path='./dataset/Descriptors/'
VII_data='VII_SMILES.csv'

df=pd.read_pickle(load_path+VII_data)
VII_SMILES=df['SMILES']

Morgan = [morgan_binary_features_generator(smi) for smi in VII_SMILES]
df_MF = pd.DataFrame(np.array(Morgan, int)) 

Column=[("MF"+str(i+1)) for i in range(0,MORGAN_NUM_BITS)]   
df_MF.columns=Column

df_MF.to_csv(save_path+"VII_Descriptors_morgan.csv")



import ipdb
import pandas as pd  
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# https://mattermodeling.stackexchange.com/questions/8541/how-to-compute-the-synthetic-accessibility-score-in-python
import sascorer

path = './viscosity_result/'

# Define a function to calculate SA score for each SMILES string
def calculate_SA(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Return None for invalid SMILES
    return sascorer.calculateScore(mol)
 
# Read and process 313K data sets 
files_313 = [path+'polyinfo_Viscosity_313.csv', path+'PI_Viscosity_313.csv', path+'Polyimides_Viscosity_313.csv']  
dfs_313 = [pd.read_csv(file) for file in files_313]  
df_313 = pd.concat(dfs_313, ignore_index=True)  
#df_313['313K_visco'] = df_313[['RF', 'KRR', 'MLP', 'XGboost']].mean(axis=1).round(3)  
df_313['313K_visco'] = df_313[['RF']].mean(axis=1).round(3) 
df_313_result = df_313[['SMILES', '313K_visco']]  
df_313_result.to_csv(path+'313K_visco.csv', index=False)  
  
# Read and process 373K data sets  
files_373 = [path+'polyinfo_Viscosity_373.csv', path+'PI_Viscosity_373.csv', path+'Polyimides_Viscosity_373.csv']  
dfs_373 = [pd.read_csv(file) for file in files_373]  
df_373 = pd.concat(dfs_373, ignore_index=True)  
#df_373['373K_visco'] = df_373[['RF', 'KRR', 'MLP', 'XGboost']].mean(axis=1).round(3) 
df_373['373K_visco'] = df_373[['RF']].mean(axis=1).round(3) 
df_373_result = df_373[['SMILES', '373K_visco']]  
df_373_result.to_csv(path+'373K_visco.csv', index=False)  

#ipdb.set_trace() # ipdb debug break point
  
# Merge the 313K and 373K data sets and calculate the PVI(Proportional Viscosity Index)  
df_merge = pd.merge(df_313_result, df_373_result, on='SMILES')  
df_merge['PVI'] = 261.1 * (df_merge['373K_visco'] ** 1.4959) / df_merge['313K_visco']
df_merge['PVI'] = df_merge['PVI'].round(3)  

# calculate the TE(Thickening Efficiency) 
viscosity_poe7 = 3.964
concentration_vii = 0.096
df_merge['TE'] = (df_merge['373K_visco'] - viscosity_poe7) / (concentration_vii * viscosity_poe7)
df_merge['TE'] = df_merge['TE'].round(3)

# Add SA column to the merged dataframe
df_merge['SA'] = df_merge['SMILES'].apply(calculate_SA).round(3)

df_merge = df_merge[['SMILES', '313K_visco', '373K_visco', 'SA', 'TE', 'PVI']]  
df_merge.columns = ['SMILES', '313K_visco', '373K_visco', 'SA', 'TE', 'PVI']  
#df_merge.to_csv(path+'screen_result.csv', index=False)
df_merge.to_csv(path+'model_performance/RF_screen_result.csv', index=False)
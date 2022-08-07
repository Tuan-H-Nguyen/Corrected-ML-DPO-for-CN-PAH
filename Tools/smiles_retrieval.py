#%%
import pandas as pd
from mo_retrieval import mol_name
import os
import glob

all_files = glob.glob(os.getcwd() + "/*.smi")
df = {"No":[], "smiles":[]}
for file in all_files:
    with open(file,'r') as o:
        line = o.readline()
        smiles = line.split("\t")[0]
        name = mol_name(file)
        df["No"].append(name)
        df["smiles"].append(smiles)

dataframe = pd.DataFrame(df)

dataframe.to_csv("cyano_smiles.csv")
# %%

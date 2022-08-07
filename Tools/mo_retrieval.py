#%%
import pandas as pd
import os
import glob
import numpy as np

data_files = glob.glob(os.getcwd()+"/*.out")

def frontier_mo(file):
    with open(file,'r') as o:
        lines = o.readlines()
        HOMO = 0 ; LUMO= 0
        for i,line in enumerate(lines):
            if "Alpha  occ. eigenvalues" in line:
                HOMO = line.split("--")[-1].split(" ")[-1].strip()
            if "Alpha virt. eigenvalues" in line and "Alpha  occ. eigenvalues" in lines[i-1]:
                LUMO = line.split("--")[-1].strip().split(" ")[0]
    return float(HOMO), float(LUMO)

def mol_name(file):
    file_parse = file.split("\\")[-1].split(".")
    name = file_parse[0]
    for char in file_parse[1]:
        if char not in [str(i) for i in range(0,10)]:
            return name
    return name + "." + file_parse[1]


dataframe = {"No":[], "IP":[], "EA":[],"BG":[]}
for file in data_files:
    HOMO,LUMO = frontier_mo(file)
    name = mol_name(file)
    dataframe["IP"].append(-HOMO*27.21)
    dataframe["EA"].append(-LUMO*27.21)
    dataframe["No"].append(name)
    dataframe["BG"].append((LUMO - HOMO)*27.21)

dataframe = pd.DataFrame(dataframe)

dataframe.to_csv("cyano_raw_data.csv")
# %%

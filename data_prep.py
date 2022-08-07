#%%
from collections import Counter
import numpy as np
import pandas as pd
from MolGraph.poly_rings.DPO import DPO_generate
from utility.logic import intersection
from substituent import PolyRingGraphWSub, AugDPO

def assign_sp(smiles):
    graph = PolyRingGraphWSub(smiles)
    segs = graph.segments
    longest = max(map(len, segs))
    longest = list(filter(
        lambda x: len(x)==longest,segs))
    
    subs = graph.substituent
    
    sa_flag = True
    result = []
    for subs in subs:
        if sa_flag and any(list(map(lambda x: x.isIn(subs.ring),longest))):
            sa_flag = False
            result.append("sa")
        else:
            result.append("s")
    return "+".join(result)

def get_stats(smiles):
    if isinstance(smiles,str):
        graph = PolyRingGraphWSub(smiles)
        return len(graph.cycles), len(graph.substituent)
    elif isinstance(smiles,list):
        num_rings = []
        num_subs = []
        for smile in smiles:
            print(smile)
            r,s = get_stats(smile)
            num_rings.append(r)
            num_subs.append(s)
        return num_rings, num_subs

def prep_dpo(smiles):   
    aug_dpo = AugDPO(overlayer_effect=False)
    aug_dpo2 = AugDPO(
        overlayer_effect=True,
        size_effect = True
        )
    aug_dpo3 = AugDPO(
        overlayer_effect=True,
        )
    
    trun_dpo = smiles.apply(
        DPO_generate
    )
    adv_sdpo = smiles.apply(
        aug_dpo2.map
    )
    adv_sdpo1 = smiles.apply(
        aug_dpo3.map
    )
    sdpo = smiles.apply(
        aug_dpo.map
    )
    
    return trun_dpo, adv_sdpo, adv_sdpo1, sdpo

def prep_data(raw_data,smiles,supplement=None):
    if isinstance(raw_data,str):
        raw_data = pd.read_csv(
            raw_data,index_col = "No").drop(
                "Unnamed: 0",axis=1
            )

    if isinstance(smiles,str):
        smiles = pd.read_csv(
            smiles,index_col = "No").drop(
                "Unnamed: 0",axis=1
            )
    
    data = pd.concat(
        [raw_data,smiles],axis=1
    ).dropna()
    
    if supplement:
        supp = pd.read_csv(
            supplement,index_col="ID").rename(
                {"No":"ID"}
            ).drop(["BG","EA","IP"],axis = 1)

        inter = intersection(
            list(raw_data.index),list(supp.index))

        data = pd.concat(
            [data.loc[inter,:],
            supp.loc[inter]],
            axis = 1
        )
    
    num_ring,num_subs = get_stats(
        list(data.loc[:,"smiles"])
    )

    data.loc[:,"no_ring"] = num_ring
    data.loc[:,"no_subs"] = num_subs

    trun_dpo,adv_sdpo,adv_sdpo1,sdpo = prep_dpo(
        data.loc[:,"smiles"]
    )

    data.loc[:,"adv_SDPO"] = adv_sdpo
    data.loc[:,"adv_SDPO1"] = adv_sdpo1
    data.loc[:,"SDPO"] = sdpo
    data.loc[:,"DPO"] = trun_dpo
    
    data = data.rename({"PAH DPO":"Full_DPO"},axis = 1)

    return data

# %%

test_data = prep_data(
    raw_data = "CyanoDATA\\Test\\raw_data.csv",
    smiles = "CyanoDATA\\Test\\smiles.csv",
    supplement="CyanoDATA\\Test\\master.csv"
    )

train_data = prep_data(
    raw_data = "CyanoDATA\\Training\\raw_data.csv",
    smiles = "CyanoDATA\\Training\\smiles.csv",
    supplement="CyanoDATA\\Training\\master.csv"
    )

misc = prep_data(
    raw_data = "CyanoDATA\\2-4CN\\raw_data.csv",
    smiles = "CyanoDATA\\2-4CN\\smiles.csv"
    )

#%%
total_data = pd.concat(
    [test_data,train_data,misc]
)

total_data.to_csv("total_data.csv")

# %%
"""
thien_data = prep_data(
    raw_data = pd.concat([
        pd.read_csv("PAH_DATA\\thien_train_set.csv",index_col="No"),
        pd.read_csv("PAH_DATA\\thien_test_set.csv",index_col="No")
    ]),
    smiles = pd.read_csv("PAH_DATA\\thienoacene_smiles.csv",index_col="No")
)

pah_data = prep_data(
    raw_data = pd.concat([
        pd.read_csv("PAH_DATA\\pah_train_set_v2.csv",index_col="No"),
        pd.read_csv("PAH_DATA\\pah_test_set.csv",index_col="No")
    ]),
    smiles = pd.read_csv("PAH_DATA\\PAH_smiles.csv",index_col="No")
)

thien_data = thien_data.rename(
    {"DPO equation":"Full_DPO"},axis= 1)
pah_data = pah_data.rename(
    {"DPO equation":"Full_DPO"},axis = 1)

"""


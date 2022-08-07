#%%
import time
from typing import Type
import numpy as np
import pandas as pd

from data_structures.cycles import CycleFinding, Cycle
from data_structures.molecular_graph import MolecularGraph
from data_structures.essential_data import ALPHABET, NUMBER
from data_structures.last_resort import sort_cycle,BruteSearchRing

start = time.time()

pah_smiles_data = pd.read_csv("_DATA\\PAH_smiles.csv",index_col="id")
thieno_smiles_data = pd.read_csv("_DATA\\thienoacene_smiles.csv",index_col="id")
smiles_data = pd.concat([
    pah_smiles_data, thieno_smiles_data
])

def ring_counter(smiles):
    counter = 0
    s_counter = 0
    for char in smiles:
        if char in NUMBER:
            pass
        elif char == "(" or char == ")":
            pass
        elif char == "c":
            counter += 1
        elif char == "s":
            s_counter += 1
        else:
            raise Exception("Unexpected character: ",char)
    return (counter -2 + 2*s_counter)/4


for i in smiles_data.index:
    smiles = smiles_data.loc[i]["smiles"]
    try:
        graph = MolecularGraph()
        graph.from_smiles(smiles)
        graph.find_cycles()#mode = "minimal")
    except : print(smiles,i)
    
    no_rings = ring_counter(smiles)
    if no_rings != len(graph.cycles):
        print("Mismatch: ",no_rings,"!=",len(graph.cycles))
        print(smiles)
    for cycle in graph.cycles:
        if len(cycle.atoms)  == 6 or len(cycle.atoms)  == 5:
            pass
        else:print("mismatch ring size",smiles)

print("Execution time: ", time.time() - start)

#%%
smiles_data = pd.read_csv("nitro_smiles.csv")
for i in smiles_data.index:
    smiles = smiles_data.loc[i]["smiles"]
    try:
        graph = MolecularGraph()
        graph.from_smiles(smiles)
        graph.find_cycles()
    except : print(smiles,i)

# %%
smiles = "s1ccc2c1c1ccccc1c1cc3cc4c(cc3cc21)c1c(cccc1)c1c4ccs1"
graph = MolecularGraph(smiles)
# %%
cf = CycleFinding()
cycles = cf.find_cycle(graph.tree.root.inner.head)
# %%
def f(node):
    ll = node.inner
    print(ll)
    for node in ll:
        print(node.value,len(node.connectivity))

graph.tree.traverse(f)
# %%
def total_number_atom(sample):
    n = 0
    for char in sample:
        if char.title() in PERIODIC_TABLE.keys() and char !="H":
            n += 1
    return n

class Test:
    def __init__(self):
        self.n = 0
    def f(self,x):
        self.n += 1
    def f_print(self,x):
        self.n += 1
        print(x, x.mark)


for i,smiles in enumerate(data.loc[:,"SMILES"]):
    test = Test()
    graph = MolecularGraph()
    graph.from_smiles(smiles)
    graph.traverse(test.f)
    n = total_number_atom(tokenize(smiles)[0])
    if test.n != n:
        print(i,smiles,test.n,"!=",n)
# %%
test = Test()
graph = MolecularGraph()
graph.from_smiles("OC1=C(C(C2=C(O)[C@@](C(C(C(N)=O)=C(O)[C@H]3N(C)C)=O)(O)[C@@]3([H])C[C@]2([H])[C@@]4(O)C)=O)C4=CC=C1")
graph.traverse(test.f_print)
# %%
graph.find_cycles()
graph.cycles
# %%

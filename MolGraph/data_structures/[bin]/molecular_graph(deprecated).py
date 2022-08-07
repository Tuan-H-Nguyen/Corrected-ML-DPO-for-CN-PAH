#%%
import os, sys

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import *
from data_structures.tree import Tree
from data_structures.cycles import CycleFinding, Cycle
from data_structures.last_resort import sort_cycle,BruteSearchRing

class FindPairsWithSameNumber:
    def __init__(self):
        self.number2AtomNode = {}

    def f(self,node):
        j = -1
        substring = node()
        for char in substring:
            if char not in NUMBER and char in ALPHABET: 
                j += 1 #pass
            elif char in NUMBER:
                try:
                    match = self.number2AtomNode[char]
                    node.inner[j].add_connection(match,1)
                    del self.number2AtomNode[char]

                except KeyError:
                    self.number2AtomNode[char] = node.inner[j]

class MolecularGraph:
    def __init__(self):
        self.tree = None
        self.cycles = None

    def from_smiles(self,smiles):
        self.tree = Tree(smiles)
        fpwsn = FindPairsWithSameNumber()
        self.tree.traverse(f = fpwsn.f)

    def find_cycles(self,mode = None):
        cf = CycleFinding()
        self.cycles = cf.find_cycle(self.tree.root.inner.head)
        single,compound = sort_cycle(self.cycles)
        if compound and mode != "minimal":
            bs = BruteSearchRing(single)
            for ring in compound:
                bs.brute_search(ring)
            self.cycles = bs.single_ring
#%%
"""sample = "C12=CC=CC=C1C3=C(C=CC=C3)C4=C2C=CC=C4"
graph = MolecularGraph()
graph.from_smiles(sample)


def f(node):
    ll = node.inner
    print(ll)
    for node in ll:
        print(node.value,len(node.connectivity))

graph.tree.traverse(f)"""

# %%

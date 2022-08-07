#%%
import os, sys, copy
import numpy as np

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)


from data_structures.molecular_graph import MolecularGraph
from data_structures.cycles import CycleFinding
from data_structures.last_resort import decompose_rings, sort_cycle

# %%
class DFS_list_edges:
    def __init__(self,allowed_nodes=None):
        self.visited_nodes = []
        self.allowed_nodes = allowed_nodes
        self.edges = []

    def traverse(self,node,prev_node = None):
        if self.allowed_nodes and node not in self.allowed_nodes:
            return
        if prev_node == None:
            pass
        else:
            edge = sorted(
                [prev_node,node],
                key = lambda x:x.random_repr)
            if edge not in self.edges:
                self.edges.append(edge)
        if node in self.visited_nodes:
            return
        self.visited_nodes.append(node)
        for atom in node.connectivity.keys():
            self.traverse(atom,node)

def ring2byte(cycle,all_edges):
    cycle_edge = []
    cycle = cycle.atoms
    for i in range(-1,len(cycle)-1):
        edge = [cycle[i],cycle[i+1]]
        edge = sorted(
            edge,key = lambda x:x.random_repr)
        cycle_edge.append(edge)
    return [int(edge in cycle_edge) for edge in all_edges]

def byte2ring(byte,all_edges):
    atoms = []
    for i,j in enumerate(byte):
        if j == 1:
            atoms += all_edges[i]
    atoms = list(set(atoms))
    cf = CycleFinding(atoms)
    cycles = cf.find_cycle(atoms[0])
    return cycles

def nand(a,b):
    return int(not (a and b))

def xor(a,b):
    return int(nand(nand(a,nand(a,b)),nand(b,nand(a,b))))

def XOR(bytes1,bytes2):
    return list(
        map(xor,bytes1,bytes2)
    )

# %%
sample = "C12=CC=CC=C1C3=C(C=CC=C3)C4=C2C=CC=C4"

graph = MolecularGraph()
graph.from_smiles(sample)
graph.find_cycles(mode = "minimal")

single_rings , compound_rings = sort_cycle(
    graph.cycles
)

list_edge = DFS_list_edges()
list_edge.traverse(graph.tree.root)
all_edges = copy.copy(list_edge.edges)


single_ring_bytes = []
for cycle in single_rings:
    single_ring_bytes.append(
        ring2byte(cycle,all_edges))

compound_ring_bytes = []
for cycle in compound_rings:
    compound_ring_bytes.append(
        ring2byte(cycle,all_edges))

compound_ring_bytes = sorted(
    compound_ring_bytes,key=np.sum
    )
# %%
ring_byte = compound_ring_bytes[1]
for sbyte in single_ring_bytes:
    new_ring_byte = XOR(ring_byte,sbyte)
    if sum(new_ring_byte) < sum(ring_byte) and sum(new_ring_byte) != 0:
        ring_byte = new_ring_byte
    else:pass
# %%
byte2ring(ring_byte,all_edges)
# %%

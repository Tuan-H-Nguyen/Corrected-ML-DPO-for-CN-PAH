#%%
import numpy as np
import os, sys

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)
#%%
class CycleFinding:
    """
    Wrapper class for classical Deep First Search 
    (DFS) for finding all rings. This method employ the
    DFS method to visit nodes and marks visited nodes. 
    Once the DFS traversal reaches a visited node, it 
    trace back the traversal path to reach the visited 
    node => ring

    Note that to use this method, the complete graph 
    has to be determined. In other word, all the edge has
    to be determined.

    It is unclear if there are any case where there are 
    undetected rings. To rule out that case, one edge of each 
    detected rings should be removed, and run the DFS 
    again. 
    Methods:
    + __init__(allowed_node = None)
    + find_cyle(start_node) -> list<Cycle> 
    """
    def __init__(self,allowed_nodes = None):
        self.cycle_list = []
        self.visited_nodes = []

        self.ring_flag = False
        self.forbidden_edge = {}
        self.counter = 0

        self.allowed_nodes = allowed_nodes

    def update_and_reset(self):
        for cycle in self.cycle_list:
            _list = cycle[0:2]
            _list = [tuple(_list),tuple(_list[::-1])]
            for v1,v2 in _list:
                try: 
                    self.forbidden_edge[v1] += [v2]
                except KeyError:
                    self.forbidden_edge.update({v1:[v2]})
        self.visited_nodes = []
        self.ring_flag = False

    def traverse(self,node,prev_node=None,path = None):
        path = path if path != None else []
        if node in self.visited_nodes:
            if len(path) >= 2 and node == path[-2]:
                #for situation where traversal return 
                #from node it has visit in last turn
                #e.g 1 -> 2 , 2 -> 1
                return
            elif node in path:
                #ring found, trace back all visited node 
                #until current node is reached
                i = path.index(node)
                self.cycle_list.append(path[i:])
                for atom in path[i:]:
                    atom.in_cycles = True
                self.ring_flag = True
                return
            else: return
        self.visited_nodes.append(node) #mark visited node
        new_path = path + [node]        #include visited node in path
        #traverse to adjacent node
        adj_nodes = list(node.connectivity.keys())
        for adj_node in adj_nodes:
            if node in self.forbidden_edge.keys() and adj_node in self.forbidden_edge[node]:
                continue
            if self.allowed_nodes != None and adj_node not in self.allowed_nodes:
                continue
            self.traverse(adj_node,node,new_path)

    def find_cycle(self,start_node):
        self.traverse(start_node)
        while self.ring_flag:
            self.counter += 1
            self.update_and_reset()
            self.traverse(start_node)
        self.cycle_list = [
            Cycle(cycle) for cycle in self.cycle_list
        ]
        return self.cycle_list

class Cycle:
    def __init__(
        self,atoms
        ):
        assert isinstance(atoms,list)
        self.atoms = atoms
        hash_code = sorted([
            atom.random_value for atom in self.atoms
        ])
        hash_code = "".join([str(num) for num in hash_code])
        self.hash_repr = str(hash(hash_code))
        self.fused_rings = {}

    def __len__(self):
        return len(self.atoms)

    def __getitem__(self,idx):
        return self.atoms[idx]

    def __repr__(self):
        return "Cycle("+str(len(self))+","+self.hash_repr+")"
    

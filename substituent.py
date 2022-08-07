#%%
import os, sys
import numpy as np
import pandas as pd

from itertools import chain, cycle

import MolGraph.__init__
from MolGraph.poly_rings.rings import FusedRing, PolyRingGraph
from MolGraph.poly_rings.segments import Segment, FindSegment
from MolGraph.poly_rings.DPO import DPO_generate
from MolGraph.poly_rings.orientation import update_segments_orientation

from MolGraph.data_structures.node import Node
from MolGraph.data_structures.molecular_graph import MolecularGraph
from MolGraph.utils.logic import intersection


class Traversal:
    def __init__(
            self,
            visited_nodes = None,
            function = None):
        self.visited_nodes = visited_nodes if visited_nodes != None else []
        self.function = function if function != None else lambda x: x

    def traverse(self,node):
        if node in self.visited_nodes:
            return
        self.visited_nodes.append(node)
        self.function(node)
        for next_node in node.connectivity:
            self.traverse(next_node)

class Substituent():
    def __init__(self,node=None,root = None):
        self.atoms = []
        if node and root:
            self.trace_sub(node,root)
        self.ring = None

    def add_node(self,node):
        self.atoms.append(node)

    def trace_sub(self,node,root):
        traversal = Traversal(function = self.add_node, visited_nodes = [root])
        traversal.traverse(node)

class PolyRingGraphWSub(PolyRingGraph):
    def __init__(self,smiles):
        super().__init__(smiles)
        self.substituent = []
        for ring in self.cycles:
            subs = self.identify_subs(ring)
            for sub in subs:
                sub.ring = ring
            self.substituent += subs

    def identify_subs(self,ring):
        subs_list = []
        fused_atoms = list(chain(*ring.fused_rings.values()))
        for atom in ring.atoms:
            if len(atom.connectivity) < 3:
                continue
            if atom in fused_atoms:
                continue
            for sub_atom in atom.connectivity:
                if sub_atom not in ring.atoms:
                    break
            sub = Substituent(node=sub_atom,root=atom)
            subs_list.append(sub)
        return subs_list

    def get_longest_len(self):
        return max([len(seg) for seg in self.segments])

class AugDPO:
    def __init__(self,overlayer_effect = False,size_effect = False):
        self.of = overlayer_effect
        self.sf = size_effect
    
    def map(self,smiles):
        graph = PolyRingGraphWSub(smiles)
        
        if len(graph.substituent) == 0:
            return "0"

        longest_len = graph.get_longest_len()
            
        result = []
        for n,segment in enumerate(graph.segments):
            if len(segment) == longest_len:
                result.append(self.subs_from_one_ref(segment,graph))
        
        result = [x for x in result if x!=[]]
        result = ["+".join(i) for i in result]
        result = ["("+i+")" for i in result]
        
        div = len(result)
        
        result = "+".join(result)
        result = "(" + result + ")"
        result = result + "*(1/{})".format(div)
        
        return result

    def subs_from_one_ref(self,segment,graph):
        result = []
        ref_sub = 0
        for subs in graph.substituent:
            
            overlayer = get_overlayer(subs,segment)
            
            if segment.isIn(subs.ring):
                ref_sub += 1
            
            elif overlayer > 0 and self.of:
                result.append("s*ds**{}".format(overlayer))
                                    
            else:
                if self.of:
                    result.append("sm")
                else:
                    result.append("s")

        if ref_sub:
            if self.sf:
                if len(segment) == 2:
                    result.append("{}".format(
                        ref_sub))
                else:
                    result.append("{}*sa{}".format(
                        ref_sub,len(segment)))
            elif self.of:
                result.append("{}-sa*({})".format(
                    ref_sub,len(segment)))
            else:
                result.append("sa*{}".format(
                    ref_sub))
        
        return result
     
def get_overlayer(sub,ref_seg):   
    oriens = ["120","60","0",None]
    for orien in oriens:
        if orien == None: 
            return -1

        adj_seg = ref_seg.adjacents[orien]
        
        adj_bool = list(map(
                lambda x:x.isIn(sub.ring),
                adj_seg))
        
        if any(adj_bool):
            break

    seg = adj_seg[adj_bool.index(True)]

    if orien == "60" or orien == "0":
        mid_seg = intersection(
            seg.adjacents["120"],
            ref_seg.adjacents["120"])[0]
        overlayer = len(mid_seg) - 1
        if orien == "0":
            return overlayer
        
    elif orien == "120":
        mid_seg = ref_seg
        overlayer = 0

    mid_ring = intersection(mid_seg.cycles,seg.cycles)[0]

    if mid_ring == sub.ring:
        return overlayer
    
    next_ring = list(filter(
        lambda x: x in seg.cycles,
        mid_ring.fused_rings.keys()))[0]
    overlayer += 1
    
    visited_rings = [mid_ring,next_ring]
    
    while next_ring != sub.ring:
        next_ring = list(filter(
            lambda x: x not in visited_rings,
            next_ring.fused_rings.keys()))
        assert len(next_ring) == 1
        next_ring = next_ring[0]
        overlayer += 1
        
    return overlayer


# %%
#"""
sample1 = "c1cc2c(cc1)c1cc3cc4cc5c(ccc6c(c(ccc56)C#N)C#N)cc4c(c3cc1c(c2)C#N)C#N"

sample2 = "N#CC1=C(C=C(C=CC2=C3C=C4C(C=CC=C4C#N)=C2)C3=C5)C5=CC=C1"

sample3 = "N#CC1=C2C=C(C=CC3=C4C5=C(C(C#N)=CC=C5)C6=C3C=CC=C6)C4=CC2=C(C=C(C=CC7=C8C=CC=C7C#N)C8=C9)C9=C1"

sample4 = "c1cc2c(cc1)c1c(c(cc3ccccc13)C#N)cc2"

sample = sample2

sdpo_gen = AugDPO(
    overlayer_effect=True,
    size_effect=True)

print(sdpo_gen.map(sample))

from rdkit import Chem

mol = Chem.MolFromSmiles(sample)

mol
#"""


# %%

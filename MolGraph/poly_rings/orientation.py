#%%
import numpy as np
import os, sys, itertools

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import *
from data_structures.cycles import CycleFinding

from poly_rings.segments import Segment

from utils.logic import intersection

def overlap(segment1,segment2):
    if isinstance(segment2,Segment):
        inter = intersection(segment1.cycles,segment2.cycles)
        assert len(inter) <= 1
        if inter and len(inter[0]) != 5:
            segment1.adjacents["120"].append(segment2)
            segment2.adjacents["120"].append(segment1)
    elif isinstance(segment2,list):
        for seg in segment2:
            overlap(segment1,seg)
    return True

def check_orientation(seg1,seg2):
    if seg1 == seg2:
        return None
    seg1_adj = seg1.adjacents["120"]
    seg2_adj = seg2.adjacents["120"]
    seg3 = intersection(seg1_adj,seg2_adj)

    if seg1 not in seg2_adj and seg2 not in seg1_adj and seg3:
        pass
    else:
        return None
    assert len(seg3) == 1
    seg3 = seg3[0]


    fused_atom_pairs = []

    #find the ring that is the overlap of each 
    # segment in the pair and the segment between them 
    seg1_3_overlap = intersection(seg1.cycles,seg3.cycles)
    assert len(seg1_3_overlap) == 1
    seg1_3_overlap = seg1_3_overlap[0]

    for ring,pair in seg1_3_overlap.fused_rings.items():
        if ring in seg1.cycles:
            fused_atom_pairs.append(pair)

    seg2_3_overlap = intersection(seg2.cycles,seg3.cycles)
    assert len(seg2_3_overlap) == 1
    seg2_3_overlap = seg2_3_overlap[0]

    for ring,pair in seg2_3_overlap.fused_rings.items():
        if ring in seg2.cycles:
            fused_atom_pairs.append(pair)

    forbidden_edge = []
    for i,cycle in enumerate(seg3.cycles[1:]):
        for ring ,atom_pair in cycle.fused_rings.items():
            if ring == seg3.cycles[i]:
                forbidden_edge.append(atom_pair)

    cf = CycleFinding(
        allowed_nodes=list(
            itertools.chain(*[cycle.atoms for cycle in seg3.cycles]))
        )

    for edge in forbidden_edge:
        atom1, atom2 = edge[0],edge[1]
        cf.forbidden_edge[atom1] = [atom2]
        cf.forbidden_edge[atom2] = [atom1]

    stitched_cycle = cf.find_cycle(
        seg3.cycles[0].atoms[0]
        )
    stitched_cycle = stitched_cycle[0].atoms

    fused_pair_index = []
    for pair in fused_atom_pairs:
        fused_pair_index.append(list(map(
            lambda x: stitched_cycle.index(x),
            pair)))
    result = []
    for i in fused_pair_index:
        if abs(i[0] - i[1]) == 1:
            result.append(min(i))
        else:
            result.append(-1)
    #print(fused_pair_index)
    #print(result)
    result = abs(result[0]-result[1])
    if result == len(stitched_cycle)/2:
        return "0"
    else:
        return "60"

def update_segments_orientation(graph):
    for segment in graph.segments:
        segment.adjacents["120"] = []
        segment.adjacents["60"] = []
        segment.adjacents["0"] = []

    for i,seg in enumerate(graph.segments):
        overlap(seg,graph.segments[i+1:])

    for i,seg in enumerate(graph.segments):
        for seg2 in graph.segments:
            orien = check_orientation(seg,seg2)
            if not orien:
                continue
            else:
                seg.adjacents[orien].append(seg2)
# %%
"""
#TEST SECTION

from poly_rings.rings import PolyRingGraph

#sample = "C12=CC=CC=C1C3=C(C=CC=C3)C4=C2C=CC=C4"
#sample = "C12=CC=C3C(C(C=CC4=C5C=CC=C4)=C5C6=C3C7=C(C=CC=C7)C=C6)=C1C=CC=C2"
#sample = "C12=CC=CC=C1C=C3C(C=CC=C3)=C2"
#sample = "C12=C(C=CS3)C3=C(C=C(C(C=CS4)=C4C5=C6C=CC7=C5C=CC=C7)C6=C8)C8=C1C(C=CC=C9)=C9C=C2"
sample = "C12=CC=C3C(C(C=CC4)=C4C5=C3C6=C(C=CC=C6)C7=C5C=CC8=C7C=C9C(C=CC9)=C8)=C1CC=C2"
graph = PolyRingGraph(sample)

update_segments_orientation(graph)

for seg in graph.segments:
    print("###############")
    for cyc in seg.cycles:
        if len(cyc) == 5:
            print(55555)
    print(seg)
    print(seg.adjacents)
"""

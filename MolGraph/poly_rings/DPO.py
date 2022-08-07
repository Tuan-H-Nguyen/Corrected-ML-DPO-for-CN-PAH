#%%
import os, sys,copy

from poly_rings.rings import PolyRingGraph

from utils.logic import intersection

def check_thiophene(segment):
    for ring in segment.cycles:
        if len(ring) == 5:
            return True

def reference_segment(segment,thiophene=False):
    n = len(segment) - 1
    a = 0
    for i in range(n):
        a += i
    if thiophene:
        return ["{} - {}*a - af".format(n,a)]
    else: return ["{} - {}*a".format(n,a)]

def b_segment(segment):
    n = len(segment) - 1
    thiophene = check_thiophene(segment)
    if thiophene and n == 1:
        return []
    overlayer = -1
    result = []
    d = "df" if thiophene else "d"
    for i in range(n):
        overlayer += 1
        result.append("b*{}**{}".format(d,overlayer))
    return result

def c_segment(
    segment,main_segment
    ):
    n = len(segment) - 1
    thiophene = check_thiophene(segment)
    if thiophene and n == 1:
        return []

    b_seg = intersection(
        segment.adjacents["120"],
        main_segment.adjacents["120"]
        )
    assert len(b_seg) == 1
    b_seg = b_seg[0]
    overlayer = len(b_seg) - 1
    d = "df" if thiophene else "d"

    result = []
    for i in range(n):
        result.append("c*{}**{}".format(d,overlayer-1))
        overlayer += 1
    return result

def d_segment(
    segment,main_segment
    ):
    b_seg = intersection(
        segment.adjacents["120"],
        main_segment.adjacents["120"]
        )
    assert len(b_seg) == 1
    b_seg = b_seg[0]
    overlayer = len(b_seg) - 1
    thiophene = check_thiophene(segment)
    d = "df" if thiophene else "d"

    result = []
    result.append(
        "(" + reference_segment(segment)[0] + ")" 
        + "*{}**{}".format(d,overlayer))    
    return result

def DPO_generate(smiles):
    DPO_list = []
    graph = PolyRingGraph(smiles)
    segments = sorted(
        graph.segments,key=len,reverse=True)
    highest_len = len(segments[0])
    for seg in segments:
        if len(seg) == highest_len:
            dpo = []
            dpo += reference_segment(
                seg,thiophene=check_thiophene(seg))
            for seg2 in seg.adjacents["120"]:
                dpo += b_segment(seg2)
            for seg2 in seg.adjacents["60"]:
                dpo += c_segment(seg2,seg)
            for seg2 in seg.adjacents["0"]:
                dpo += d_segment(seg2,seg)
            DPO_list.append(" + ".join(dpo))
    DPO_list = ["(" + dpo + ")" for dpo in DPO_list]
    divider = "*(1/{})".format(len(DPO_list))
    DPO_list = "(" + " + ".join(DPO_list) + ")"
    return DPO_list + divider
"""
sample = "c(sc1c2)(cc(c(c3)c4)cc5c4cccc5)c3c1cc6c2cccc6"
DPO_generate(sample)
"""
# %%

# %%

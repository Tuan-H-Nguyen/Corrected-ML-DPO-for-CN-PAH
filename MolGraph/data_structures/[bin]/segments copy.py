#%%
import os, sys,copy

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import *

class Segment:
    def __init__(self,cycles=None):
        self.cycles = [] if cycles == None else cycles
        self.adjacents = {}
        self.hash_repr = None

    def __len__(self):
        return len(self.cycles)

    def __repr__(self):
        if not self.hash_repr:
            hash_code = sorted([
                cycle.hash_repr for cycle in self.cycles
            ])
            self.hash_repr = hash("".join(hash_code))
        return "Segment("+str(len(self))+","+str(self.hash_repr)+")"

class FindSegment:
    def __init__(self):
        self.segments = []
        self.visited_ring = []

    def find_single_segment(
        self,ring,segment=None
        ):
        if ring in self.visited_ring:
            return
        first_flag = False
        if segment == None:
            first_flag = True
            new_segment = Segment([ring]) 
        
        self.visited_ring.append(ring)

        next_rings = ring.fused_rings.keys()
        if first_flag:
            pass
        else:
            new_segment = copy.deepcopy(segment)
            new_segment.cycles.append(ring)
            if len(next_rings) == 3:
                self.segments.append(new_segment)
                new_segment = Segment([ring])
            elif len(next_rings) == 2:
                pass
            elif len(next_rings) == 1:
                self.segments.append(new_segment)
                return
        for next_ring in next_rings:
            self.find_single_segment(
                next_ring,new_segment)
    
    def find_segments(self,cycles):
        for i,cycle in enumerate(cycles):
            cycle.find_fused_ring(cycles[i:])
            if len(cycle.fused_rings.keys()) == 3:
                root = cycle
        print(root)
        self.find_single_segment(root)
        return self.segments

# %%

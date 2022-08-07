from copy import deepcopy
import os, sys

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.essential_data import ALPHABET, BOND_CHARACTER, LOWER_ALPHABET, NUMBER, UPPER_ALPHABET
from data_structures.node import Node, AtomNode


class LinkedList:
    """
    Linked List data structure is used to model linear chain of
    atom between the parentheses in the SMILES string. 
    Linked list is a chain of nodes that each node contains its 
    own data and the reference to the next node in the chain.
    Arguments:
    + sub_smiles (string): string of linear chain of atoms
    """
    def __init__(self,node_class=AtomNode):
        self.len = 0
        self.head = None
        self.rear = None
        self.node = node_class


    def from_smiles(self,sub_smiles):
        self.value = sub_smiles
        if sub_smiles[0] in BOND_CHARACTER.keys():
            self.prefix = BOND_CHARACTER[sub_smiles[0]]
            self.len += 1
        elif sub_smiles[0] in UPPER_ALPHABET:
            self.prefix = 1
        elif sub_smiles[0] in LOWER_ALPHABET:
            self.prefix = "AROMATIC"
        else:
            raise Exception("Unexpected symbol: "+sub_smiles[0])

        self.head = self.node(sub_smiles[self.len])
        self.len += 1
        prev_atom = self.head

        current_atom = None
        bond_flag = 1
        for char in sub_smiles[self.len:]:
            if char in ALPHABET:
                current_atom = self.node(char)
                if prev_atom.value in LOWER_ALPHABET and char in LOWER_ALPHABET:
                    bond_flag = "AROMATIC"
                current_atom.add_connection(
                    prev_atom,bond_flag)
                prev_atom.next = current_atom
                bond_flag = 1 #consume the bond flag
                prev_atom = current_atom
                self.len += 1
            elif char in NUMBER:
                pass
            elif char == "=":
                bond_flag = 2
            elif char == "#":
                bond_flag = 3
            else:
                raise Exception("Unexpected Symbol: "+char)
        self.rear = current_atom if current_atom else self.head

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        if isinstance(idx,slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        elif isinstance(idx,int):
            current_atom = self.head
            if idx >= len(self):
                raise IndexError
            if idx < 0:
                idx = len(self) + idx
            while idx > 0:
                current_atom = current_atom.next
                idx -= 1
            return current_atom
        else: raise TypeError("Invalid Argument")

    def __iter__(self):
        self.idx = 0
        self.current_item = self.head
        return self

    def __next__(self):
        if self.idx < len(self):
            return_item = self.current_item
            self.current_item = self.current_item.next
            self.idx += 1
            return return_item
        else:
            raise StopIteration

    def __repr__(self):
        return "LinkedList("+self.value+")"

    def add_connection(self,other_linked_list):
        """
        Add connection between the rear of a linked list
        with the head of another linked list
        """
        self.rear.add_connection(
            other_linked_list.head,
            other_linked_list.prefix)

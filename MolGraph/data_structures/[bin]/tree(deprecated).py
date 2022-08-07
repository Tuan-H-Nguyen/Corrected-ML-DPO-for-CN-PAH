#%%
import re
import numpy as np
import os, sys

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from data_structures.node import AtomNode, Node
from data_structures.linked_list import LinkedList
from data_structures.essential_data import NUMBER

class TreeNode:
    """
    Class for node in tree.
    """
    def __init__(
        self,inner_class = LinkedList,
        node_class = AtomNode
        ):
        self.inner = None
        self.string = None
        self.children = []
        self.inner_class = inner_class
        self.node_class = node_class

    def from_smiles(self,smiles):
        self.string = smiles
        self.inner = self.inner_class(self.node_class)
        self.inner.from_smiles(smiles)

    def __call__(self):
        return self.string

    def __repr__(self):
        return "TreeNode(" + self.random_repr + ")"

    def add_child(self,other_tree_node):
        """
        Adding connection between node of tree is actually
        adding connection between linked lists represented by
        the node tree, which is actually adding connection 
        between the atom in the rear of one linked list to 
        the head of another linked list.
        """
        self.children.append(other_tree_node)
        self.inner.add_connection(other_tree_node.inner)


class Tree:
    """
    Class for the tree. The tree has each node as the 
    linear chain of atoms between parethesis in the SMILES
    string. For example n_1(n_2)n_3, if n_1 is root, n_2 is left 
    of n_1 and n_3  is left of n_1. The root of the tree is 
    the first term in the SMILES string.

    Note: the tree can be constructed back to the SMILES string by
    post order traversing. Algorithm for traverse:
        1. Visit the current node.
        2. Visit the node that is left of the current node.
        3. Visit the node that is right of the current node.
    Also note that this algorithm can be recursively implemented.
    
    This class also has implemention for Pre Order Traversing 
    in the tree. Each node is visit and applied (not implemented) 
    function f. To use, write a function f that take node as input,
    and call self.traverse(f).
    """
    def __init__(
        self, smiles,
        inner_class=LinkedList,
        node_class=AtomNode
        ):
        self.inner_class = inner_class
        self.node_class = node_class

        smiles_list = re.split("(\(|\))", smiles)
        self.root = TreeNode(self.inner_class,self.node_class)
        self.root.from_smiles(smiles_list[0])
        self.trace_tree(smiles_list,self.root,1)

    def traverse(self,f):
        traverse(self.root,f)

    def trace_tree(
        self,nodes_list,prev_node,
        current_idx):
        """
        Recursively create the tree from 
        n_1(n_2(n_3)n_4(n_5))n_6... where n_i is the i-th 
        sub-string of the SMILES string. 
        """
        while current_idx < len(nodes_list):
            if nodes_list[current_idx] == '(':
                #move to the next node
                current_idx += 1
                #which should be a chain of atoms and can be turned into a LL
                curr_node = TreeNode(self.inner_class,self.node_class)
                curr_node.from_smiles(nodes_list[current_idx])
                #set new node as the left branch of the current node
                prev_node.add_child(curr_node)
                #move to the next node
                current_idx += 1
                #recursively call the function to continue from the
                #new node, which then return the index from which 
                #the current function can be continued.
                current_idx = self.trace_tree(
                    nodes_list,curr_node,current_idx)
            elif nodes_list[current_idx] == ')':
                #end the function
                current_idx += 1
                return current_idx
            elif nodes_list[current_idx] == "":
                current_idx += 1
            else:
                curr_node = TreeNode(self.inner_class,self.node_class)
                curr_node.from_smiles(nodes_list[current_idx])
                #add connection btw the rear and head of 2 LinkedLists
                prev_node.add_child(curr_node)
                prev_node = curr_node
                current_idx += 1


def traverse(root,f):
    if root:
        f(root)
        for child in root.children:
            traverse(child,f)

# %% TEST
"""
smiles = "s1ccc2c1c1ccccc1c1cc3cc4c(cc3cc21)c1c(cccc1)c1c4ccs1"
treexample = Tree(smiles)
treexample.traverse(f=lambda x:print(x()))
"""
#for i in treexample.root.inner:
#    print(i)

#re.split("(\(|\))", smiles)

# %%

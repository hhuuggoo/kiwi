import tree_helper

#sub-indexes in this module should be numpy arrays of integers

class FieldDescriptor():
    def __init__(self, name, discrete_int):
        self.name=name
        self.discrete_int=discrete_int

class Tree:
    def __init__(self, data, target, data_descriptors, target_descriptor, sub_indexes):
        self.data = data
        self.target = target
        self.data_descriptors = data_descriptors
        self.target_descriptor = target_descriptor
        self.discreteout = discreteout
        self.flds = flds


class SimpleTreeNode:
    def __del__(self):
        del self.sub_indexes
        del self.level
        for x in self.children:
            del x
    def __init__(self, tree, sub_indexes, level, parent_node):
        self.tree = tree
        self.sub_indexes = sub_indexes
        self.level = level
        self.parent_node = parent_node
        self.children=[]
        self.rules=[]

def simple_tree_eval(root_node,data):
    

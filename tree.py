"""
Tree.py
FieldDescriptor
    ContinuousField
    DiscreteField
Rule
    DiscreteBinaryRule
    ContinuousBinaryRule
Tree
SimpleBinaryTreeNode
"""

import numpy as np
import tree_helper as th
class FieldDescriptor:
    """ Describes a column of your data """
    def __init__(self, name, discrete, col_idx):
        """
        name: string
        discrete: boolean
        col_idx: number
        """
        self.name = name
        self.col_idx = col_idx
        self.discrete = discrete


class ContinuousField(FieldDescriptor):
    """ Describes continuous data"""
    def __init__(self, name, col_idx):
        FieldDescriptor.__init__(self, name, False, col_idx)


class DiscreteField(FieldDescriptor):
    """ Describes discrete data"""
    def __init__(self, name, col_idx, unique_values):
        """
        name: string
        discrete: boolean
        col_idx: int
        unique_values: numpy array of floats
        """
        FieldDescriptor.__init__(self, name, True, col_idx)
        self.unique_values = unique_values
        self.discrete_val_mapping = None
    def assign_labels(self, label_list):
        """
        label_list: list of strings
        """
        assert (len(label_list) == len(self.unique_values)), \
               "label_list wrong length"
        self.discrete_val_mapping = {}
        for c in range(len(label_list)):
            self.discrete_val_mapping[self.unique_values[c]] = label_list[c]
            
class ContinuousTarget(ContinuousField):
    def __init__(self, name):
        ContinuousField.__init__(self, name, 0)
        del self.col_idx
        
    
class Rule:
    def __init__(self, field, value):
        self.field = field
        self.value = value
    def evaluate_rule(self, sample_data):
        raise NotImplementedError, "this is an interface"

class DiscreteBinaryRule(Rule):
    def evaluate_rule(self, sample_data):
        sample_value = sample_data[self.field.col_idx]
        if sample_value == self.value:
            return 1
        else:
            return 0
class ContinuousBinaryRule(Rule):
    def evaluate_rule(self, sample_data):
        sample_value = sample_data[self.field.col_idx]
        if sample_value > self.value:
            return 1
        else:
            return 0
class Tree:
    def __init__(self, data, target, data_descriptors, target_descriptor):
        self.data = np.array(data, 'float64')
        self.target = np.array(target, 'float64')
        self.data_descriptors = data_descriptors
        self.target_descriptor = target_descriptor
        self.root = None
        self.metric_code = None
        self.metric_state = None
        self.metric_output = None
        self.stop_code = None
        self.output_code = None
        
    def predict(self, sample):
        if self.root is not None:
            return self.root.descend(sample).output
        
    def helper_data_init(self):
        self.disc_array = np.array([x.discrete for x in self.data_descriptors])
        self.unique_vals_list = []
        for x in self.data_descriptors:
            if x.discrete:
                self.unique_vals_list.append(x.unique_values)
            else:
                self.unique_vals_list.append(None)
        
    def grow(self, sub_idx, store_data = False):
        self.helper_data_init()
        self.root = SimpleBinaryTreeNode(self, 0, None, store_data)
        self.root.grow(sub_idx)
        
        
class SimpleBinaryTreeNode:
    def __init__(self, tree, level, parent_node, store_data):
        self.tree = tree
        self.level = level
        self.parent_node = parent_node
        self.store_data = store_data
        self.stat_store = []
        self.rule = None
        self.output = None
        self.children = []

    def descend(self, sample):
        if len(self.children) == 0:
            return self
        else:
            idx = self.rule.evaluate_rule(sample)
            if idx == 0:
                return self.children[0].descend(sample)
            else:
                return self.children[1].descend(sample)

    def grow(self, sub_idx):
        self.output = self.tree.output_func(self, sub_idx)
        
        if self.tree.stop_func(self, sub_idx) or len(sub_idx)<2:
            return
        
        (col_idx, val, score, idx1, idx2) = th.split(self.tree.data[sub_idx, :],
                                                     self.tree.target[sub_idx],
                                                     self.tree.disc_array,
                                                     self.tree.unique_vals_list,
                                                     self.tree.metric_state,
                                                     self.tree.metric_output)
        field = self.tree.data_descriptors[col_idx]
        
        if self.store_data:
            self.stat_store.append((val, score, sub_idx[idx1], sub_idx[idx2]))
        if field.discrete:
            self.rule = DiscreteBinaryRule(field, val)
        else:
            self.rule = ContinuousBinaryRule(field, val)
            
        self.children.append(SimpleBinaryTreeNode(self.tree, self.level+1,
                                                  self, self.store_data))
        self.children.append(SimpleBinaryTreeNode(self.tree, self.level+1,
                                                  self, self.store_data))
        self.children[0].grow(idx1)
        self.children[1].grow(idx2)


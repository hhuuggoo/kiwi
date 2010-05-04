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
        assert (len(label_list) == len(self.unique_values)), "label_list wrong length"
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
        self.data = data
        self.target = target
        self.data_descriptors = data_descriptors
        self.target_descriptor = target_descriptor
        self.root = None
        
    def predict(self, sample):
        if self.root is not None:
            return self.root.descend(sample).output
        
    def grow(self, sub_idx, metric_func, stop_func,
             output_func, store_data = False):
        self.root = SimpleBinaryTreeNode(self, 0, None, store_data)
        self.root grow(self, sub_idx, metric_func,
                       stop_func, output_func)
        
        
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

    def grow(self, sub_idx, metric_func, stop_func, output_func):
        self.output = output_func(self, sub_idx)
        
        if stop_func(self, sub_idx) or len(sub_idx)<2:
            return
        
        (field, val, score, idx1, idx2) = self.split(sub_idx, metric_func)
        
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
        self.children[0].grow(idx1, metric_func, stop_func, output_func)
        self.children[1].grow(idx2, metric_func, stop_func, output_func)

    def split(self, sub_idx, metric_func):
        sub_data = self.tree.data[sub_idx, :]
        sub_target = self.tree.target[sub_idx]
        num_cols = sub_data.shape[1]
        best_score = None
        best_val = 0
        best_idx1 = None
        best_idx2 = None
        best_field = None
        for cc in range(num_cols):
            field = self.tree.data_descriptors[cc]
            if field.discrete:
                (val,
                 score,
                 idx1,
                 idx2) = self.split_discrete(sub_data[:,cc],
                                                  sub_target,
                                                  cc,
                                                  metric_func)
            else:
                (val,
                 score,
                 idx1,
                 idx2) = self.split_continuous(sub_data[:,cc],
                                                    sub_target,
                                                    cc,
                                                    metric_func)
            if not np.isfinite(score):
                continue
            
            if best_score is None or score > best_score:
                best_val = val
                best_score = score
                best_idx1 = idx1
                best_idx2 = idx2
                best_field = field
        return(best_field, best_val, best_score, best_idx1, best_idx2)
    
    def split_discrete(self, sub_column_data, sub_target,
                      column_idx, metric_func):
        discrete_values = self.tree.data_descriptors[column_idx].unique_values
        best_score = None
        for x in discrete_values:
            idx = (sub_column_data == x)
            not_idx = np.logical_not(idx)
            target1 = sub_target[idx]
            target2 = sub_target[not_idx]
            score = metric_func(target1, target2)
            
            if not np.isfinite(score):
                continue
            
            if best_score is None or score > best_score:
                best_score = score
                best_discrete_class = x
                best_idx = idx
                best_not_idx = not_idx
        return(best_discrete_class, best_score, np.nonzero(best_idx), np.nonzero(best_not_idx))
    
    def split_continuous(self, sub_column_data, sub_target,
                         column_idx, metric_func):
        sorted_idx = np.argsort(sub_column_data)
        sorted_target = sub_target[sorted_idx]
        best_score = None
        best_idx = 0
        for x in range(len(sorted_idx)):
            target1 = sorted_target[:x]
            target2 = sorted_target[x:]
            score = metric_func(target1, target2)
            if not np.isfinite(score):
                continue
            
            if best_score is None or score > best_score:
                best_score = score
                best_idx = x
        best_value = sorted_idx[best_idx]
        idx1 = sorted_idx[:best_idx]
        idx2 = sorted_idx[best_idx:]
        return (best_value, best_score, idx1, idx2)

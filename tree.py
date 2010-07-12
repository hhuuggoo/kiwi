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
import tree_func_c as tf_c

numstr = "%.4f"

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

        #functions that determine tree behavior
        #if code is set, use funcs implied by the code, else
        #use the funcs that are passed in
        
        self.metric_code = tf_c.NO_METRIC
        self.metric_state = None
        self.metric_output = None

    def enable_all(self, node = -1):
        if node == -1:
            node = self.root
        node.terminate = False
        for n in node.children:
            self.enable_all(node = n)
            
    def prune(self, stop_func, node = -1):
        if node == -1:
            node = self.root
            
        if stop_func(node, node.stat_store['idx1'], node.stat_store['idx2']):
            for n in node.children:
                n.terminate = True
        else:
            for n in node.children:
                self.prune(stop_func, node = n)
            
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
        self.root.compute_output(sub_idx)
        self.root.grow(sub_idx)


        
        
class SimpleBinaryTreeNode:
    def __init__(self, tree, level, parent_node, store_data):
        self.tree = tree
        self.level = level
        self.parent_node = parent_node
        self.store_data = store_data
        self.stat_store = {}
        self.rule = None
        self.output = None
        self.children = []
        self.terminate = False
        
    def descend(self, sample):
        if len(self.children) == 0 or self.terminate:
            return self
        else:
            idx = self.rule.evaluate_rule(sample)
            if idx == 0:
                return self.children[0].descend(sample)
            else:
                return self.children[1].descend(sample)
            
    def compute_output(self, sub_idx):
        self.output = self.tree.output_func(self, sub_idx)
        
    def grow(self, sub_idx):
        (col_idx, val, score, idx1, idx2) = tf_c.split(self.tree.data[sub_idx, :],
                                                       self.tree.target[sub_idx],
                                                       self.tree.disc_array,
                                                       self.tree.unique_vals_list,
                                                       self.tree.metric_state,
                                                       self.tree.metric_output,
                                                       self.tree.metric_code)
        field = self.tree.data_descriptors[col_idx]
        sub_idx1 = sub_idx[idx1]
        sub_idx2 = sub_idx[idx2]

                    
        if self.store_data:
            self.stat_store['sub_idx'] = sub_idx
            self.stat_store['val'] = val
            self.stat_store['score'] = score
            self.stat_store['idx1'] = sub_idx[idx1]
            self.stat_store['idx2'] = sub_idx[idx2]

        if (self.tree.stop_func(self, sub_idx1, sub_idx2) or
            len(sub_idx1) < 2 or len(sub_idx2) < 2):
            return
            
        if field.discrete:
            self.rule = DiscreteBinaryRule(field, val)
        else:
            self.rule = ContinuousBinaryRule(field, val)
            
        self.children.append(SimpleBinaryTreeNode(self.tree, self.level+1,
                                                  self, self.store_data))
        self.children[0].compute_output(sub_idx1)
        self.children.append(SimpleBinaryTreeNode(self.tree, self.level+1,
                                                  self, self.store_data))
        self.children[1].compute_output(sub_idx2)
        self.children[0].grow(sub_idx1)
        self.children[1].grow(sub_idx2)

def node_verify(tree, node):
    def process_idx(idx, verif_val):
        wrong_list = []
        for i in idx:
            data = tree.data[i,:]
            if node.rule.evaluate_rule(data) != verif_val:
                wrong_list.append(i)
        return wrong_list
    
    idx1 = node.stat_store['idx1']
    idx2 = node.stat_store['idx2']
    verif_val = 0
    wrong1 = process_idx(idx1, 0)            
    wrong2 = process_idx(idx2, 1)
    return (wrong1, wrong2)


            

"""
output funcs for crude tree visualization
"""

default_func_list = [lambda x: x.output,
                     lambda x: x.rule.field.name,
                     lambda x: x.rule.value]

def tree2txt(tree, fname, func_list = default_func_list, sep = ""):
    data_mat = arrayNode(tree.root, func_list)['data']
    data_mat = np.rot90(data_mat)
    col_max_width = []
    for cc in range(data_mat.shape[1]):
        max_width = 0
        for rc in range(data_mat.shape[0]):
            data = data_mat[rc,cc]
            if data not in ["=","|"]:
                if type(data) is str:
                    new_width = len(data)
                if new_width > max_width:
                    max_width = new_width
        col_max_width.append(max_width)
    f = open(fname, 'w')
    for rc in range(data_mat.shape[0]):
        for cc in range(data_mat.shape[1]):
            data = data_mat[rc,cc]
            if data == "=":
                data = data * col_max_width[cc]
            elif type(data) is str:
                data = data.ljust(col_max_width[cc])
            data_mat[rc,cc] = data
        output = sep.join([str(x) for x in data_mat[rc,:]])
        f.write(output + "\n")
    f.flush()
    f.close()
    
def arrayNode(node, func_list):
    small_data = np.empty((len(func_list), 1), 'object')
    for idx, f in enumerate(func_list):
        try:
            small_data[idx,0] = f(node)
        except:
            small_data[idx,0] = None
        if type(small_data[idx,0]) is str:
            small_data[idx,0] = '(%s)' % small_data[idx,0]
        elif small_data[idx,0] is None:
            small_data[idx,0] = "()"
        else:
            small_data[idx,0] = '(%s)' % (numstr % small_data[idx,0])

    if len(node.children)==0 or node.terminate:
        return {'data':small_data, 'w1':0, 'w2':0}
    else:
        children_data = [arrayNode(x, func_list) for x in node.children]
        small_height = small_data.shape[0]
        child_height1 = children_data[0]['data'].shape[0]
        child_height2 = children_data[1]['data'].shape[0]
        child_height = np.max((child_height1, child_height2))
        
        big_height = small_height + child_height
        width1 = children_data[0]['data'].shape[1]
        width2 = children_data[1]['data'].shape[1]
        joined_data = np.tile("", (big_height, width1 + width2 + 1))
        joined_data = np.array(joined_data, 'object')
        joined_data[:small_height, [width1]] = small_data
        joined_data[small_height:(small_height + child_height1), :width1] = children_data[0]['data']
        joined_data[small_height:(small_height + child_height2), -width2:] = children_data[1]['data']

        mid_pt = np.ceil(len(func_list)/2)
        joined_data[mid_pt, children_data[0]['w1']:width1] = "|"
        joined_data[mid_pt:small_height, children_data[0]['w1']] = "="
        joined_data[mid_pt, -width2:(-children_data[1]['w2']-1)] = "|"
        joined_data[mid_pt:small_height, (-children_data[1]['w2']-1)] = "="
        
    return {'data':joined_data,
            'w1':width1,
            'w2':width2}


        

import numpy as np

def max_depth(threshold):
    def helper(tree_node, sub_idx1, sub_idx2):
        return tree_node.level + 1 > threshold
    return helper

def md_min_subtree(threshold, num_elements):
    def helper(tree_node, sub_idx1, sub_idx2):
        return ((tree_node.level >= threshold) or
                (len(sub_idx1) < num_elements) or
                len(sub_idx2) < num_elements)
    return helper

def mean_output():
    def helper(tree_node, sub_idx):
        return np.mean(tree_node.tree.target[sub_idx])
    return helper

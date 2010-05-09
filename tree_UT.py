import numpy as np
from tree import *
import tree_func_c
import tree_func

num_rows=2000
num_cols=11
discretes = [4,5]
data = np.random.random((num_rows,num_cols))
unique_vals = np.arange(-10.0,11.0)
unique_labels = [str(x) for x in unique_vals]
for c in discretes:
    data[:,c] = np.round(10 * data[:,c])

target = np.random.random(num_rows)
target_desc = ContinuousTarget('target') 


fld_desc=[]
for c in range(num_cols):
    if c in discretes:
        descriptor = DiscreteField(str(c), c, np.array(unique_vals))
        descriptor.assign_labels(unique_labels)
    else:
        descriptor = ContinuousField(str(c), c)
    fld_desc.append(descriptor)

assert fld_desc[discretes[0]].discrete_val_mapping[5.0] == '5.0', 'discrete mapping value wrong'


disc_rule = DiscreteBinaryRule(fld_desc[discretes[0]], 5.0)
disc_sample = np.zeros(num_cols)
disc_sample[discretes[0]] = 5.0
assert disc_rule.evaluate_rule(disc_sample) == 1, 'discrete rule wrong'
disc_sample[discretes[0]] = 1.0
assert disc_rule.evaluate_rule(disc_sample) == 0, 'discrete rule wrong'

cont_rule = ContinuousBinaryRule(fld_desc[0], 0.1)
cont_sample = np.zeros(num_cols)
cont_sample[0] = 0.0
assert cont_rule.evaluate_rule(cont_sample) == 0, 'cont rule wrong'
cont_sample[0] = 1.0
assert cont_rule.evaluate_rule(cont_sample) == 1, 'cont rule wrong'

my_tree = Tree(data, target, fld_desc, target_desc)
root = SimpleBinaryTreeNode(my_tree, 0, None, True)
root.output = np.mean(target)
root.rule = ContinuousBinaryRule(fld_desc[0], 0.0)
my_tree.root = root


child1 = SimpleBinaryTreeNode(my_tree, 1, None, True)
child1.output = -1.0

child2 = SimpleBinaryTreeNode(my_tree, 1, None, True)
child2.output = 1.0

root.children.append(child1)
root.children.append(child2)

cont_sample = np.zeros(num_cols)
cont_sample[0] = 0.5

assert my_tree.predict(cont_sample) == 1.0, 'prediction wrong'

def max_depth(tree_node, sub_idx):
    return tree_node.level > 10

def mean_output(tree_node, sub_idx):
    return np.mean(tree_node.tree.target[sub_idx])
    
root.children = []
root.output = None
root.rule = None

import datetime as dt
my_tree.helper_data_init()
my_tree.metric_state = tree_func_c.mse_metric_state
my_tree.metric_output = tree_func_c.mse_metric_output
my_tree.output_func = tree_func.mean_output()
my_tree.stop_func = tree_func.max_depth(10)
st=dt.datetime.today()
root.grow(np.arange(num_rows))
ed=dt.datetime.today()
print ed-st

print my_tree.predict(cont_sample) 




import tree_func_c
import numpy as np
import datetime as dt
target = np.array(np.arange(20000),'float64')
data = np.array(target<100, 'float64')

st=dt.datetime.today()
result = tree_func_c.split_continuous(data, target, tree_func_c.mse_metric_state, tree_func_c.mse_metric_output)
ed=dt.datetime.today()
print(ed-st)
print result

import tree_func_c as tf_c
data = np.array([1,2,3,4,5,6,7,8,9,10], 'float64')
state = tf_c.mse_metric_state([], data, 0.0,  True, True)
output = tf_c.mse_metric_output([state])

print state, output

state1 = tf_c.mse_metric_state(state, data[:5], 0.0,  True, False)
state2 = tf_c.mse_metric_state([], data[:5], 0.0,  True, True)
output = tf_c.mse_metric_output([state, state2])

print state1, state2, output

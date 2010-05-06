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

import tree_helper
import numpy as np
import datetime as dt
target = np.array(np.arange(20000),'float64')
data = np.array(target<100, 'float64')

st=dt.datetime.today()
result = tree_helper.tree_split_column_continuous(data, target, tree_helper.mse_metric_c)
ed=dt.datetime.today()
print(ed-st)
print result

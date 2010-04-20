import tree_helper


class Tree:
    def __init__(self, data, target_idx, discreteout, flds):
        total_indexes = range(data.shape[1])
        del total_indexes[target_idx]
        self.data = data[:,total_indexes]
        self.target = [:,target_idx]
        self.discreteout = discreteout
        self.flds = flds


class SimpleTreeNode:
    def __del__(self):
        del self.sub_data
        del self.sub_indexes
        del self.sub_target
        del self.level
        del self.parent
        for x in self.children:
            del x
    def __init__(self, sub_data, sub_indexes, sub_target, level, parent):
        self.sub_data=sub_data
        self.sub_indexes=sub_indexes
        self.sub_target=sub_target
        self.level=level
        self.parent=parent
        self.children=[]
    

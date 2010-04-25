import numpy as np
cimport numpy as np

cdef class FieldDescriptor:
    cdef public object name
    cdef public int data_column_index
    cdef public bint discrete_int
    def __cinit__(self, object name, bint discrete_int, int data_column_index):
        self.discrete_int=discrete_int
        self.data_column_index = data_column_index
    def __init__(self, object name, bint discrete_int, int data_column_index):
        self.name=name
        
cdef class ContinuousField(FieldDescriptor):
    def __cinit__(self, object name, bint discrete_int, int data_column_index):
        FieldDescriptor.__cinit__(self, name, discrete_int, data_column_index)
        
    def __init__(self, object name, bint discrete_int, int data_column_index):
        FieldDescriptor.__init__(self, name, discrete_int, data_column_index)

cdef class DiscreteField(FieldDescriptor):
    cdef public object discrete_val_mapping
    cdef public np.ndarray unique_values
    def __cinit__(self, object name, bint discrete_int, int index,
                  np.ndarray unique_values, object labels):
        FieldDescriptor.__cinit__(self,name,discrete_int,index)
        self.unique_values=unique_values
        
    cpdef assign_labels(self, object label_list):
        self.discrete_val_mapping={}
        cdef int c, length
        for c in range(length):
            self.discrete_val_mapping[self.unique_values[c]] = label_list[c]
            
cdef class Rule:
    cdef public FieldDescriptor field
    cdef public double value
    cdef object statistics_storage
    def __cinit__(self, FieldDescriptor field, double value):
        self.field=field
        self.value=value
    cpdef bint evaluate_rule(self, np.ndarray sample_data):
        return 0
    
cdef class DiscreteBinaryRule(Rule):
    cpdef bint evaluate_rule(self, np.ndarray sample_data):
        cdef np.ndarray[np.float64_t, ndim=1] sample = sample_data
        cdef double sample_value = sample_data[self.field.index]
        if sample_value == self.value:
            return 1
        else:
            return 0
        
cdef class ContinuousBinaryRule(Rule):
    cpdef bint evaluate_rule(self, np.ndarray sample_data):
        cdef np.ndarray[np.float64_t, ndim=1] sample = sample_data
        cdef double sample_value = sample_data[self.field.index]
        if sample_value > self.value:
            return 1
        else:
            return 0


cdef class Tree:
    cdef public np.ndarray data
    cdef public np.ndarray target
    cdef public object data_descriptors
    cdef FieldDescriptor target_descriptor
    
    def __cinit__(self, np.ndarray data, np.ndarray target, object data_descriptors, FieldDescriptor target_descriptor):
        self.data = data
        self.target = target
        self.target_descriptor = target_descriptor
        
    def __init__(self,  np.ndarray data, np.ndarray target, object data_descriptors, FieldDescriptor target_descriptor):
        self.data_descriptors = data_descriptors
        
cdef class SimpleBinaryTreeNode:
    cdef public Tree tree
    cdef public int level
    cdef public SimpleBinaryTreeNode parent_node
    cdef public bint store_data
    cdef object children
    cdef Rule rule
    
    def __cinit__(self, Tree tree, int level,
                  SimpleBinaryTreeNode parent_node, bint store_data):
        self.tree = tree
        self.level = level
        self.parent_node = parent_node
        self.store_data = store_data
        
    def __init__(self, Tree tree, int level,
                 SimpleBinaryTreeNode parent_node, bint store_data):
        self.children=[]
        
    cpdef SimpleBinaryTreeNode descend(self, np.ndarray sample):
        cdef bint idx
        if len(self.children)==0:
            return self
        else:
            idx = self.rule.evaluate_rule(sample)
            if idx:
                return self.children[0].descend(sample)
            else:
                return self.children[1].descend(sample)
            
    #Currently, we train using python functions, however we might take
    #codes which determine which c functions to use later, for speed optimizations
    
    cpdef train(self, object metric_func=None, object stopping_func=None, int store_data=0):
        cdef np.ndarray[np.int_t, ndim=1] idx1, idx2
        if not stopping_func(self):
            self.rule = tree_split(self, metric_func)
            
cpdef object tree_split(SimpleBinaryTreeNode node, metric_func, int store_data):    
    cdef np.ndarray[np.float64_t, ndim=2] data,target
    data=node.tree.data[node.sub_indexes,:]
    target=node.tree.target[node.sub_indexes]
    cdef np.ndarray[np.int_t, ndim=1] idx1, idx2
    cdef int num_cols,cc,rc
    cdef double best_score,score,val,best_val
    cdef int best_column
    best_score=0.0
    num_cols = data.shape[1]
    for cc in range(num_cols):
        if node.tree.data_descriptors[cc].discrete_int:
            (val,score,idx1,idx2)=tree_split_column_discrete(data[:,cc], target, metric_func)
        else:
            (val,score,idx1,idx2)=tree_split_column_continuous(data[:,cc],target,metric_func)
        if score>best_score:
            score=best_score
            best_column=cc
            best_val=val
    cdef FieldDescriptor best_field = node.tree.data_desciptors[cc]
    if best_field.discrete_int:
        return (DiscreteBinaryRule(best_field, best_val), idx1, idx2)
    else:
        return (ContinuousBinaryRule(best_field, best_val), idx1, idx2)
    

cpdef object tree_split_column_discrete(np.ndarray input_data,
                                        np.ndarray input_target, metric_func):
    cdef np.ndarray[np.float64_t, ndim=2] data,target
    data=input_data
    target=input_target
    cdef np.ndarray[np.float64_t, ndim=1] discrete_values, target1, target2
    cdef np.ndarray idx, best_idx
    cdef double x,score,best_score, best_discrete_class
    best_score=0.0
    discrete_values = np.unique1d(data)
    for x in discrete_values:
        idx = (data==x)
        target1=target[idx]
        target2=target[np.logical_not(idx)]
        score = metric_func(target1,target2)
        if score>best_score:
            score=best_score
            best_discrete_class = x
            best_idx = idx
    return (best_discrete_class, score, np.nonzero(best_idx), np.nonzero(np.logical_not(best_idx)))

cpdef object tree_split_column_continuous(np.ndarray input_data,
                                          np.ndarray input_target, metric_func):
    
    cdef np.ndarray[np.float64_t, ndim=1] target1, target2, sorted_target,data,target
    data=input_data
    target=input_target
    cdef np.ndarray[np.int_t, ndim=1] sorted_idx
    cdef double score,best_score, best_value
    cdef int x, idxlen best_idx
    sorted_idx = np.argsort(data)
    sorted_target = target[sorted_idx]
    idxlen=len(sorted_idx)
    best_score=0.0
    for x in range(idxlen):
        target1 = sorted_target[:x]
        target2 = sorted_target[x:]
        score = metric_func(target1,target2)
        if score>best_score:
            score=best_score
            best_value = data[sorted_idx[x]]
            best_idx = x
    return (best_value, best_score, sorted_idx[:best_idx], sorted_idx[best_idx:])

cdef double mse_metric_c(np.ndarray tdata1,
                    np.ndarray tdata2):
    cdef np.ndarray[np.float64_t, ndim=1] data1 = tdata1
    cdef np.ndarray[np.float64_t, ndim=1] data2 = tdata2
    cdef int i,data_length
    cdef double accume_square_val, accume_sum_val,error1,error2
    data_length=len(data1)
    accume_square_val=0.0
    accume_sum_val=0.0
    for i in range(data_length):
        accume_sum_val +=data1[i]
    accume_sum_val = accume_sum_val/data_length
    for i in range(data_length):
        accume_square_val += (data1[i]-accume_sum_val)**2
    error1 = accume_square_val/data_length

    data_length=len(data2)
    accume_square_val=0.0
    accume_sum_val=0.0
    for i in range(data_length):
        accume_sum_val +=data2[i]
    accume_sum_val = accume_sum_val/data_length
    for i in range(data_length):
        accume_square_val += (data2[i]-accume_sum_val)**2
    error2 = accume_square_val/data_length\

    return (error1+error2)

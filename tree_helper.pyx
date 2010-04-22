import numpy as np
cimport numpy as np

def tree_spit(np.ndarray[np.float64_t, ndim=2] data,
              np.ndarray[np.float64_t, ndim=1] target,
              discrete_indicator_list, metric_func):
    cdef int num_cols,cc,rc
    cdef double best_score,score,val,best_val
    cdef int best_column
    best_score=0.0
    num_cols = data.shape[1]
    for cc in range(num_cols):
        if discrete_indicator_list[cc]==1:
            (val,score)=tree_split_column_discrete(data[:,cc], target, metric_func)
        else:
            (val,score)=tree_split_column_continuous(data[:,cc],target,metric_func)
        if score>best_score:
            score=best_score
            best_column=cc
            best_val=val
    return(best_column,best_val,score)

def tree_split_column_discrete(np.ndarray[np.float64_t,ndim=1] data,
                               np.ndarray[np.float64_t,ndim=1] target, metric_func):
    cdef np.ndarray[np.float64_t, ndim=1] discrete_values, target1, target2
    cdef np.ndarray idx
    cdef double x,score,best_score, best_discrete_class
    best_score=0.0
    discrete_values = np.unique1d(data)
    for x in discrete_values:
        idx = (data==x)
        target1=target[idx]
        target2=target[-1 * idx]
        score = metric_func(target1,target2)
        if score>best_score:
            score=best_score
            best_discrete_class=x
    return(best_discrete_class, best_score)

def tree_split_column_continuous(np.ndarray[np.float64_t, ndim=1] data,
                                 np.ndarray[np.float64_t,ndim=1] target, metric_func):
    cdef np.ndarray[np.float64_t, ndim=1] target1, target2, sorted_target
    cdef np.ndarray[np.int_t, ndim=1] sorted_idx
    cdef double score,best_score, best_value
    cdef int x, idxlen
    sorted_idx = np.argsort(data)
    sorted_target = target[sorted_idx]
    idxlen=len(sorted_idx)
    best_score=0.0
    for x in range(idxlen):
        target1=target[:x]
        target2=target[x:]
        score = metric_func(target1,target2)
        
        if score>best_score:
            score=best_score
            best_value=data[sorted_idx[x]]
    return(best_value, best_score)

cpdef mse_metric_c(np.ndarray tdata1,
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

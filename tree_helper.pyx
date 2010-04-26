import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def  general_mse_metric_c(np.ndarray tdata1,
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
    error2 = accume_square_val/data_length

    return -(error1+error2)


@cython.boundscheck(False)
@cython.wraparound(False)
def split(np.ndarray sub_data_in, np.ndarray sub_target_in,
            np.ndarray disc_array, object unique_vals_list, metric_func):
    cdef np.ndarray[np.float64_t, ndim = 2] sub_data = sub_data_in
    cdef np.ndarray[np.float64_t, ndim = 1] sub_target = sub_target_in
    cdef np.ndarray[np.int_t, ndim=1] best_idx1, best_idx2, idx1, idx2
    cdef int num_cols, cc, best_col_idx
    cdef double best_score, score, val, best_val
    cdef bool score_set = False

    num_cols = sub_data.shape[1]

    for cc in range(num_cols):
        if disc_array[cc]:
            (val,
             score,
             idx1,
             idx2) = split_discrete(sub_data[:,cc],
                                  sub_target,
                                  unique_vals_list[cc],
                                  metric_func)
        else:
            (val,
             score,
             idx1,
             idx2) = split_continuous(sub_data[:,cc],
                                                sub_target,
                                                metric_func)
        if not np.isfinite(score):
            continue

        if not score_set or best_score > score:
            best_val = val
            best_score = score
            best_idx1 = idx1
            best_idx2 = idx2
            best_col_idx = cc
            
    return(best_col_idx, best_val, best_score, best_idx1, best_idx2)

@cython.boundscheck(False)
@cython.wraparound(False)
def split_discrete(np.ndarray sub_column_data_in, np.ndarray sub_target_in,
                   np.ndarray discrete_values_in, metric_func):
    cdef np.ndarray[np.float64_t, ndim=1] sub_column_data, sub_target, discrete_values
    sub_column_data = sub_column_data_in
    sub_target = sub_target_in
    discrete_values = discrete_values_in

    cdef bool score_set = False
    cdef double x, best_score, best_discrete_class
    cdef np.ndarray  idx, not_idx, best_idx, best_not_idx
    
    for x in discrete_values:
        idx = (sub_column_data == x)
        not_idx = np.logical_not(idx)
        target1 = sub_target[idx]
        target2 = sub_target[not_idx]
        if len(target1)==0 or len(target2)==0:
            continue
        score = metric_func(target1, target2)

        if not score_set or  score > best_score:
            score_set = True
            best_score = score
            best_discrete_class = x
            best_idx = idx
            best_not_idx = not_idx
    return(best_discrete_class, best_score, np.nonzero(best_idx)[0], np.nonzero(best_not_idx)[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def split_continuous(np.ndarray sub_column_data_in, np.ndarray sub_target_in, metric_func):
    cdef np.ndarray[np.float64_t, ndim=1] sub_column_data, sub_target, target1, target2
    cdef np.ndarray[np.int_t, ndim=1] sorted_idx, idx1, idx2
    cdef bool score_set = False
    cdef int x, idx_len, best_idx
    cdef double best_score, score, best_value
    
    sub_column_data = sub_column_data_in
    sub_target = sub_target_in
    
    sorted_idx = np.argsort(sub_column_data)
    sorted_target = sub_target[sorted_idx]
    idx_len = len(sorted_idx)

    for x from 0 < x < idx_len:
        score = metric_func(sorted_target, x)
        if not score_set or score > best_score:
            score_set = True
            best_score = score
            best_idx = x

    best_value = sorted_idx[best_idx]
    idx1 = sorted_idx[:best_idx]
    idx2 = sorted_idx[best_idx:]
    return (best_value, best_score, idx1, idx2)

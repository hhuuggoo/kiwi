import numpy as np
cimport numpy as np
import cython
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object mse_metric_state(object state, object data_in, object val_in,
                              bint use_array, bint to_add):
    
    cdef np.ndarray[np.float64_t, ndim = 1] data 
    cdef double accum_square_val, accum_sum_val, accum_length, error1, error2,
    cdef int i, data_length

    if use_array:
        data = data_in
        data_length = len(data)
        accum_square_val = 0.0
        accum_sum_val = 0.0
    
        for i in range(data_length):
            accum_sum_val += data[i]
        for i in range(data_length):
            accum_square_val += data[i]**2
    else:
        accum_square_val = val_in**2
        accum_sum_val = val_in
        data_length = 1
        
    if not to_add:
        accum_square_val = -accum_square_val
        accum_sum_val = -accum_sum_val
        data_length = -data_length
        
    if len(state) == 0:
        state = [0, 0, 0]
        
    state[0] = state[0] + accum_square_val
    state[1] = state[1] + accum_sum_val
    state[2] = state[2] + float(data_length)

    return state

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mse_metric_output(object state_list):
    #x^2 - 2ux + u^2
    cdef double total_error = 0
    cdef double total_length = 0
    cdef double accum_square_val, accum_sum_val, data_length

    for state in state_list:
        accum_square_val = state[0]
        accum_sum_val = state[1]
        data_length =  state[2]
        error = accum_square_val - \
                 2 * accum_sum_val * accum_square_val + \
                 accum_sum_val ** 2
        total_error += error
        total_length += data_length
    return total_error / total_length
        
        
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object split(np.ndarray sub_data_in, np.ndarray sub_target_in,
                   np.ndarray disc_array, object unique_vals_list,
                   metric_state, metric_output):
    
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
                                    metric_state,
                                    metric_output)
        else:
            (val,
             score,
             idx1,
             idx2) = split_continuous(sub_data[:,cc],
                                      sub_target,
                                      metric_state,
                                      metric_output)
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
cpdef object split_discrete(np.ndarray sub_column_data_in, np.ndarray sub_target_in,
                   np.ndarray discrete_values_in, metric_state, metric_output):
    
    cdef np.ndarray[np.float64_t, ndim=1] sub_column_data, sub_target, discrete_values
    sub_column_data = sub_column_data_in
    sub_target = sub_target_in
    discrete_values = discrete_values_in

    cdef bool score_set = False
    cdef double x, best_score, best_discrete_class
    cdef np.ndarray  idx, not_idx, best_idx, best_not_idx

    master_state = metric_state([], sub_target, 0.0, True, True)
    for x in discrete_values:
        idx = (sub_column_data == x)
        class_target = sub_target[idx]
        class_state = metric_state([], class_target, 0.0, True, True)
        left_state = metric_state(master_state, class_target, 0.0, True, False)
        score = metric_output([class_state, left_state])
        
        if not np.isfinite(score):
            continue

        if not score_set or  score > best_score:
            best_score = score
            best_discrete_class = x
            best_idx = idx
            best_not_idx = np.logical_not(idx)

    return(best_discrete_class, best_score, np.nonzero(best_idx)[0], np.nonzero(best_not_idx)[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object split_continuous(np.ndarray sub_column_data_in,
                              np.ndarray sub_target_in, metric_state,
                              metric_output):
    
    cdef np.ndarray[np.float64_t, ndim=1] sub_column_data, sub_target
    cdef np.ndarray[np.int_t, ndim=1] sorted_idx, idx1, idx2
    cdef bool score_set = False
    cdef int x, idx_len, best_idx
    cdef double best_score, score, best_value
    
    sub_column_data = sub_column_data_in
    sub_target = sub_target_in
    
    sorted_idx = np.argsort(sub_column_data)
    sorted_target = sub_target[sorted_idx]
    idx_len = len(sorted_idx)
    greater_state = metric_state([], sorted_target, 0.0, True, True)
    lesser_state = []
    for x in range(len(sorted_idx)):
        lesser_state = metric_state(lesser_state, sorted_target, sorted_target[x],
                                    False, True)
        greater_state = metric_state(greater_state, sorted_target, sorted_target[x],
                                     False, False)
        score = metric_output([lesser_state, greater_state])
        if not np.isfinite(score):
            continue

        if not score_set or score > best_score:
            best_score = score
            best_idx = x

    best_value = sorted_idx[best_idx]
    idx1 = sorted_idx[:best_idx]
    idx2 = sorted_idx[best_idx:]
    return (best_value, best_score, idx1, idx2)

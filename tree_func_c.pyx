#cython: profile=True

import numpy as np
cimport numpy as np
import cython
cimport cython

NO_METRIC = 0
MSE = 1
R2 = 2
cdef int NO_METRIC_C = 0
cdef int MSE_C = 1
cdef int R2_C = 2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object state_eval(object metric_state, int metric_code, object state, np.ndarray data_in, double val_in,
                        bint use_array, bint to_add):
    cdef object data
    if metric_code == NO_METRIC_C:
        data = metric_state(state, data_in, val_in, use_array, to_add)
    elif metric_code == MSE_C:
        data =  mse_metric_state(state, data_in, val_in, use_array, to_add)
    elif metric_code == R2_C:
        data =  r2_metric_state(state, data_in, val_in, use_array, to_add)
    return data

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double output_eval(object metric_output, int metric_code,  object state):
    cdef double metric
    if metric_code == NO_METRIC_C:
        metric = metric_output(state)
    elif metric_code == MSE_C:
        metric = mse_metric_output(state)
    elif metric_code == R2_C:
        metric = r2_metric_output(state)

    return metric

    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object mse_metric_state(object state, np.ndarray data_in, double val_in,
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
    cdef double mean, error

    for state in state_list:
        accum_square_val = state[0]
        accum_sum_val = state[1]
        data_length =  state[2]
        if data_length == 0:
            error = 0.0
        else:
            mean = accum_sum_val / data_length
            error = accum_square_val - \
                    2 * mean * accum_sum_val + \
                    data_length * mean ** 2
        total_error += error
        total_length += data_length
    return -total_error / total_length


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object r2_metric_state(object state, np.ndarray data_in, double val_in,
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
cpdef double r2_metric_output(object state_list):
    #x^2 - 2ux + u^2
    cdef double total_error = 0
    cdef double total_length = 0
    cdef double accum_square_val, accum_sum_val, data_length
    cdef double mean, error, global_mean, var, total_var

    accum_sum_val = 0.0
    data_length = 0.0
    for state in state_list:
        accum_sum_val += state[1]
        data_length += state[2]
    global_mean = accum_sum_val / data_length
        
    for state in state_list:
        accum_square_val = state[0]
        accum_sum_val = state[1]
        data_length =  state[2]
        if data_length == 0:
            error = 0.0
            var = 0.0
        else:
            mean = accum_sum_val / data_length
            error = accum_square_val - \
                    2 * mean * accum_sum_val + \
                    data_length * mean ** 2
            var = accum_square_val - \
                    2 * global_mean * accum_sum_val + \
                    data_length * global_mean ** 2
        total_error += error
        total_var += var

    return 1 - total_error / total_var

        
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object split(np.ndarray sub_data_in, np.ndarray sub_target_in,
                   np.ndarray disc_array, object unique_vals_list,
                   metric_state, metric_output, int metric_code):
    
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
                                    metric_output,
                                    metric_code)
        else:
            (val,
             score,
             idx1,
             idx2) = split_continuous(sub_data[:,cc],
                                      sub_target,
                                      metric_state,
                                      metric_output,
                                      metric_code)
        if not np.isfinite(score):
            continue

        if not score_set or best_score > score:
            best_val = val
            best_score = score
            best_idx1 = idx1
            best_idx2 = idx2
            best_col_idx = cc
            score_set = True
    return(best_col_idx, best_val, best_score, best_idx1, best_idx2)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object split_discrete(np.ndarray sub_column_data_in, np.ndarray sub_target_in,
                   np.ndarray discrete_values_in, metric_state, metric_output, int metric_code):
    
    cdef np.ndarray[np.float64_t, ndim=1] sub_column_data, sub_target, discrete_values
    sub_column_data = sub_column_data_in
    sub_target = sub_target_in
    discrete_values = discrete_values_in

    cdef bool score_set = False
    cdef double x, best_score, best_discrete_class, score
    cdef np.ndarray  idx, not_idx, best_idx, best_not_idx
    cdef object master_state, class_state
    master_state = state_eval(metric_state, metric_code, [], sub_target, 0.0, True, True)
    for x in discrete_values:
        idx = (sub_column_data == x)
        class_target = sub_target[idx]
        class_state = state_eval(metric_state, metric_code, [], class_target, 0.0, True, True)
        left_state = state_eval(metric_state, metric_code, master_state, class_target, 0.0, True, False)
        score = output_eval(metric_output, metric_code, [class_state, left_state])
        
        if not np.isfinite(score):
            continue

        if not score_set or  score > best_score:
            best_score = score
            best_discrete_class = x
            best_idx = idx
            best_not_idx = np.logical_not(idx)
            score_set
    return(best_discrete_class, best_score, np.nonzero(best_idx)[0], np.nonzero(best_not_idx)[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object split_continuous(np.ndarray sub_column_data_in,
                              np.ndarray sub_target_in, metric_state,
                              metric_output, int metric_code):
    
    cdef np.ndarray[np.float64_t, ndim=1] sub_column_data, sub_target
    cdef np.ndarray[np.int_t, ndim=1] sorted_idx, idx1, idx2
    cdef bool score_set = False
    cdef int x, idx_len, best_idx, length
    cdef double best_score, score, best_value
    
    sub_column_data = sub_column_data_in
    sub_target = sub_target_in
    
    sorted_idx = np.argsort(sub_column_data)
    sorted_target = sub_target[sorted_idx]
    idx_len = len(sorted_idx)
    greater_state = state_eval(metric_state, metric_code, [],
                               sorted_target, 0.0, True, True)
    lesser_state = []
    length = len(sorted_idx)
    for x in range(length):
        lesser_state = state_eval(metric_state, metric_code,
                                  lesser_state, sorted_target, sorted_target[x],
                                  False, True)
        greater_state = state_eval(metric_state, metric_code, greater_state,
                                   sorted_target, sorted_target[x],
                                   False, False)

        if x+1 < length and sorted_target[x+1] == sorted_target[x]:
            continue
        
        score = output_eval(metric_state, metric_code,
                            [lesser_state, greater_state])
        if not np.isfinite(score):
            continue

        if not score_set or score > best_score:
            best_score = score
            best_idx = x
            score_set = True
    best_value = sub_column_data[sorted_idx[best_idx]]
    idx1 = sorted_idx[:best_idx]
    idx2 = sorted_idx[best_idx:]
    return (best_value, best_score, idx1, idx2)

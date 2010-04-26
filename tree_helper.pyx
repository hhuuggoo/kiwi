import numpy as np
cimport numpy as np

cpdef double mse_metric_c(np.ndarray tdata1,
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

    return -(error1+error2)

def split(sub_data, sub_target, metric_func):
    num_cols = sub_data.shape[1]
    best_score = None
    best_val = 0
    best_idx1 = None
    best_idx2 = None
    best_field = None
    for cc in range(num_cols):
        field = self.tree.data_descriptors[cc]
        if field.discrete:
            (val,
             score,
             idx1,
             idx2) = self.split_discrete(sub_data[:,cc],
                                              sub_target,
                                              cc,
                                              metric_func)
        else:
            (val,
             score,
             idx1,
             idx2) = self.split_continuous(sub_data[:,cc],
                                                sub_target,
                                                cc,
                                                metric_func)
        if not np.isfinite(score):
            continue

        if best_score is None or score > best_score:
            best_val = val
            best_score = score
            best_idx1 = idx1
            best_idx2 = idx2
            best_field = field
    return(best_field, best_val, best_score, best_idx1, best_idx2)

def split_discrete(self, sub_column_data, sub_target,
                  column_idx, metric_func):
    discrete_values = self.tree.data_descriptors[column_idx].unique_values
    best_score = None
    for x in discrete_values:
        idx = (sub_column_data == x)
        not_idx = np.logical_not(idx)
        target1 = sub_target[idx]
        target2 = sub_target[not_idx]
        score = metric_func(target1, target2)

        if not np.isfinite(score):
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_discrete_class = x
            best_idx = idx
            best_not_idx = not_idx
    return(best_discrete_class, best_score, np.nonzero(best_idx), np.nonzero(best_not_idx))

def split_continuous(self, sub_column_data, sub_target,
                     column_idx, metric_func):
    sorted_idx = np.argsort(sub_column_data)
    sorted_target = sub_target[sorted_idx]
    best_score = None
    best_idx = 0
    for x in range(len(sorted_idx)):
        target1 = sorted_target[:x]
        target2 = sorted_target[x:]
        score = metric_func(target1, target2)
        if not np.isfinite(score):
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_idx = x
    best_value = sorted_idx[best_idx]
    idx1 = sorted_idx[:best_idx]
    idx2 = sorted_idx[best_idx:]
    return (best_value, best_score, idx1, idx2)

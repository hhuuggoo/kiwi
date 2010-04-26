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

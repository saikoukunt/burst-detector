import numpy as np
import warnings

def mk_GP_mat(T1, T2, params):

    # Input parsing
    if len(params) == 1:
        params = np.array([1, params[0], 2])
    elif len(params) == 2:
        params = np.array([params[0], params[1], 2])
    elif len(params) == 3:
        params = np.array(params)
    else:
        warnings.warn("params has too many elements. Using the first 3 only!")
        params = np.array(params[:3])

    # Generate matrix
    T1 = T1[..., np.newaxis]
    T2 = T2[..., np.newaxis]

    K = np.abs(np.tile(T1,(T2.T.shape)) - np.tile(T2.T,(T1.shape)))
    K = params[0]*np.exp(-(K**params[2])/(params[1]**params[2]))

    return K
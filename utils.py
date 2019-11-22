import numpy as np
from numpy.core._multiarray_umath import ndarray


def join_struct_arrays(arrays):
    new_dt = list(set(sum([a.dtype.descr for a in arrays], [])))
    joint = np.zeros(arrays[0].shape, dtype=new_dt)
    for a in arrays:
        for nm in a.dtype.names:
            joint[nm] = a[nm]
    return joint

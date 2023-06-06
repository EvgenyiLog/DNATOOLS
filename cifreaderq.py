import os
import numpy as np
import ctypes


def cifreader(f):
    if os.path.splitext(f)[1]==".cif":
        cluster_count=np.fromfile(f,count=9,offset=12,dtype=ctypes.c_uint)
        print()
        unknow=np.fromfile(f,count=12,offset=-1,dtype=ctypes.c_uint)
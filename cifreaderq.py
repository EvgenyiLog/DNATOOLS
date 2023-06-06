import os
import numpy as np
import ctypes


def cifreader(f):
    if os.path.splitext(f)[1]==".cif":
        cluster_count=np.fromfile(f,count=-1,offset=9,dtype=ctypes.c_ulong)#ctypes.c_uint
        print(cluster_count)
        print()
        
        
        


if __name__ == '__main__':
    cifreader("C:/Users/Евгений/Downloads/s_1_1101.cif")
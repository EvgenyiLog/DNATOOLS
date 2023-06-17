import os
import numpy as np
import ctypes


def cifreader(f):
    if os.path.splitext(f)[1]==".cif":
        cluster_count=np.fromfile(f,count=4,offset=9,dtype=ctypes.c_ulong)#ctypes.c_uint
        #print(cluster_count)
        #print(type(cluster_count))
        #print(cluster_count.shape)
        #print()
        intensitivity=np.fromfile(f,count=-1,offset=13,dtype=ctypes.c_uint16)#ctypes.c_uint
        #print(intensitivity)
        #print(type(intensitivity))
        
        #print(intensitivity.shape)
        #print(intensitivity.size)
        intensitivity=np.resize(intensitivity,(int(intensitivity.shape[0]//4),4))
        #print(intensitivity.shape)
        #print(intensitivity.size)
        intensitivityA,intensitivityC,intensitivityG,intensitivityT=np.hsplit(intensitivity, 4)
        intensitivityA=np.divide(intensitivityA,cluster_count[0])
        intensitivityC=np.divide(intensitivityC,cluster_count[1])
        
        intensitivityG=np.divide(intensitivityG,cluster_count[2])
        
        intensitivityT=np.divide(intensitivityT,cluster_count[3])
        intensitivityA=np.resize(intensitivityA,intensitivityA.size)
        intensitivityT=np.resize(intensitivityT,intensitivityT.size)
        intensitivityC=np.resize(intensitivityC,intensitivityC.size)
        intensitivityG=np.resize(intensitivityG,intensitivityG.size)
        #print(intensitivityT.shape)

        
    
        #print(intensitivityA.shape)
       
        #print(intensitivityA.max())
        #print(intensitivityA.sum())
        #print(np.sum(intensitivityA))
        #print(intensitivityC.sum())
        #print(intensitivityG.sum())
        #print(intensitivityT.sum())
        #print()
        return intensitivityA,intensitivityC,intensitivityG,intensitivityT,cluster_count
        
       
        
        
        
        


#if __name__ == '__main__':
    #intenA,intenC,intenG,intenT,quantitycluster=cifreader("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.cif")
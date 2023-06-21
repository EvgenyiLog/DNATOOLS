import os
import numpy as np
import ctypes
import matplotlib.pyplot as plt


def cifreader(f):
    if os.path.splitext(f)[1]==".cif":
        cluster_count=np.fromfile(f,count=4,offset=9,dtype=ctypes.c_uint32)#ctypes.c_uint
        #print(cluster_count)
        #print(type(cluster_count))
        #print(cluster_count.shape)
        print()
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
        intensitivityT=np.divide(intensitivityT,cluster_count[2])
        intensitivityA=np.resize(intensitivityA,intensitivityA.size)
        intensitivityC=np.resize(intensitivityC,intensitivityC.size)
        intensitivityG=np.resize(intensitivityG,intensitivityG.size)
        intensitivityT=np.resize(intensitivityT,intensitivityT.size)
        #print(intensitivityA.shape)

        plt.figure('intensitivityA',figsize=(15,7))
        plt.hist(intensitivityA,bins=50,density=True,stacked=True, facecolor='r',histtype= 'bar',edgecolor='k',linewidth=2, alpha=0.75)
        plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        

        plt.figure('intensitivityC',figsize=(15,7))
        plt.hist(intensitivityC,bins=50,density=True,stacked=True, facecolor='r',histtype= 'bar',edgecolor='k',linewidth=2, alpha=0.75)
        plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        
        plt.figure('intensitivityG',figsize=(15,7))
        plt.hist(intensitivityG,bins=50,density=True,stacked=True, facecolor='r',histtype= 'bar',edgecolor='k',linewidth=2, alpha=0.75)
        plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        
        plt.figure('intensitivityT',figsize=(15,7))
        plt.hist(intensitivityT,bins=50,density=True,stacked=True, facecolor='r',histtype= 'bar',edgecolor='k',linewidth=2, alpha=0.75)
        plt.grid(True)
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        
       
        #print(intensitivityA.max())
        #print(intensitivityA.sum())
        #print(np.sum(intensitivityA))
        #print(intensitivityC.sum())
        #print(intensitivityG.sum())
        #print(intensitivityT.sum())
        print()
        plt.show()
        return np.sum(intensitivityA),np.sum(intensitivityC),np.sum(intensitivityG),np.sum(intensitivityT),cluster_count
        
        


if __name__ == '__main__':
    #cifreader("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.cif")
    sumintenA,sumintenC,sumintenG,sumintenT,quantitycluster=cifreader("C:/Users/Евгений/Downloads/s_1_1101.cif")
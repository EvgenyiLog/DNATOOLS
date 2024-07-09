import cv2
from LOCSFile import LOCSFile
import os
import sys
import numpy as np
import pandas  as pd
from imagereader import readimage,tiffreader
import matplotlib.pyplot as plt 

def locsreader(f):
    'locs reader'
    if os.path.splitext(f)[1]==".locs":
        file=LOCSFile(f)
        z=file.read_record_locs()
        x,y=next(z)
        #print(x)
        #print(y)
        

        

        return x,y
    
import pprint
def densityclusters(filepath1,filepath2,w1=320,h1=456):
    xcentr,ycentr=locsreader(filepath1)
    image=tiffreader(filepath2)
    w,h=image.shape[:2]
    print(f'Weight={w},Height={h}')
    s=np.multiply(w,h)
    k=len(xcentr)
    densityclustersv=k/s
    pprint.pprint(densityclustersv)
    s1=w1*h1
    k1=densityclustersv*s1
    k1=np.rint(k1)
    pprint.pprint(k1)
    


def main():
    densityclusters("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.locs","C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101_c.tif")



if __name__ == '__main__':
    main()

    


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
    #assert print('File does not exist')
    return x,y



def locsintensivity(filepath1,filepath2):
    image=tiffreader(filepath1)
    x,y=locsreader(filepath2)
    intensivitylocsbf=[]
    for x,y in zip(x,y):
        intensivitylocsbf.append(image[int(np.floor(y)),int(np.floor(x))])

    d={'intensivitylocsbf':intensivitylocsbf}
    df = pd.DataFrame(data=d)
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/intensivitylocsbf.xlsx")
    writer = pd.ExcelWriter(pathsave, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/intensivitylocsbf.csv")
    df.to_csv(pathsave, index=False,encoding='utf-8',sep=';')
    print("writing complete")



def main():
    locsintensivity("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101_c.tif","C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.locs")



if __name__ == "__main__":
    main()








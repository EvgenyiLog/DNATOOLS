import os
import sys
import numpy as np
import pandas  as pd
from LOCSFile import LOCSFile

def locsreader(f):
    'locs reader'
    if os.path.splitext(f)[1]==".locs":
        file=LOCSFile(f)
        z=file.read_record_locs()
        x,y=next(z)
    return x,y

def converter(f):
    xcentr,ycentr=locsreader(f)
    d={'xcentr':xcentr,'ycentr':ycentr}
    
    df = pd.DataFrame(data=d)
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/locs.xlsx")
    writer = pd.ExcelWriter(pathsave, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/locs.csv")
    df.to_csv(pathsave, index=False,encoding='utf-8',sep=';')
    print("writing complete")



def main():
    converter("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.locs")



if __name__ == "__main__":
    main()
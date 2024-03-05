import os
import sys
import numpy as np
import pandas  as pd
from cifreaderq import cifreader
from LOCSFile import LOCSFile
from BCLFile   import BCLFile



def locsreader(f):
    'locs reader'
    if os.path.splitext(f)[1]==".locs":
        file=LOCSFile(f)
        z=file.read_record_locs()
        x,y=next(z)
    return x,y


def readerbcl(f):
    'bclreader'
    if os.path.splitext(f)[1]==".bcl":
        file=BCLFile(f)
        x=file.read_record_bcl()
        base,qual=next(x)
        #print(base)
        #print(qual)
        qual=np.subtract(qual,33)
        return base,qual
    

def converter(f1,f2,f3,filename='dnafile.xlsx',save_dir="C:/Users/evgen/Downloads/DNATOOOLS/result/",savecsv=True,savexlsx=True):
    base,qual=readerbcl(f1)
    xcentr,ycentr=locsreader(f2)
    intensitivityA,intensitivityC,intensitivityG,intensitivityT,cluster_count,cycle=cifreader(f3)
    
    d={'base':base,'quality':qual,'xcentr':xcentr,'ycentr':ycentr,'intensitivityA':intensitivityA[:len(xcentr)],'intensitivityC':intensitivityC[:len(ycentr)],'intensitivityG':intensitivityG[:len(ycentr)],'intensitivityT':intensitivityT[:len(xcentr)]}
    #d={'xcentr':xcentr,'ycentr':ycentr,'intensitivityA':intensitivityA,'intensitivityG':intensitivityG,'intensitivityC':intensitivityC,'intensitivityT':intensitivityT}
    
    df = pd.DataFrame(data=d)
    if savexlsx:
        #pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/dnafile.xlsx")
        pathsave=os.path.join(save_dir, filename)
        pathsave=os.path.splitext(os.path.abspath(pathsave))[0]+".xlsx"
        writer = pd.ExcelWriter(pathsave, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.close()
        #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
        print("writing complete")
    
    if savecsv:
        #pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/dnafile.csv")
        pathsave=os.path.join(save_dir, filename)
        pathsave=os.path.splitext(os.path.abspath(pathsave))[0]+".csv"
        df.to_csv(pathsave, index=False,encoding='utf-8',sep=';')
        print("writing complete")


def main():
    converter("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.bcl","C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.locs","C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.cif")



if __name__ == "__main__":
    main()
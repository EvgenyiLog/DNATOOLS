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
    

def converter(f1,f2,f3):
    base,qual=readerbcl(f1)
    xcentr,ycentr=locsreader(f2)
    intensitivityA,intensitivityC,intensitivityG,intensitivityT,cluster_count=cifreader(f3)
    l1=len(intensitivityA)
    l2=len(intensitivityC)
    l3=len(intensitivityT)
    l4=len(intensitivityG)
    i=np.asarray([l1,l2,l3,l4])
    #print(i)
    maxlen=np.amax(i)
    a=np.zeros_like(intensitivityA)
    #print(maxlen)
    #base=base+[0]*(maxlen - len(base))
    #qual=qual+[0]*(maxlen - len(qual))
    #qual=qual+a[len(a):]
    #base=base+a[len(a):]
    d={'base':base,'quality':qual,'xcentr':xcentr,'ycentr':ycentr,'intensitivityA':intensitivityA[:len(xcentr)],'intensitivityG':intensitivityG[:len(ycentr)],'intensitivityC':intensitivityC[:len(ycentr)],'intensitivityT':intensitivityT[:len(xcentr)]}
    #d={'xcentr':xcentr,'ycentr':ycentr,'intensitivityA':intensitivityA,'intensitivityG':intensitivityG,'intensitivityC':intensitivityC,'intensitivityT':intensitivityT}
    
    df = pd.DataFrame(data=d)
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/dnafile.xlsx")
    writer = pd.ExcelWriter(pathsave, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/dnafile.csv")
    df.to_csv(pathsave, index=False,encoding='utf-8',sep=';')
    print("writing complete")


def main():
    converter("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.bcl","C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.locs","C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.cif")



if __name__ == "__main__":
    main()
import os
import sys
import numpy as np
import pandas  as pd
from BCLFile   import BCLFile


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

def bclconverter(f):
    base,qual=readerbcl(f)
    d={'base':base,'quality':qual}
    df = pd.DataFrame(data=d)
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/bcl.xlsx")
    writer = pd.ExcelWriter(pathsave, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/bcl.csv")
    df.to_csv(pathsave, index=False,encoding='utf-8',sep=';')
    print("writing complete")



def main():
    bclconverter("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.bcl")



if __name__ == "__main__":
    main()


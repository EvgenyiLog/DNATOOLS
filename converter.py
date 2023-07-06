import os
import sys
import numpy as np
import pandas  as pd
from cifreaderq import cifreader



def converter(f):
    intensitivityA,intensitivityC,intensitivityG,intensitivityT,cluster_count=cifreader(f)
    d={'intensitivityA':intensitivityA,'intensitivityG':intensitivityG,'intensitivityC':intensitivityC,'intensitivityT':intensitivityT}
    df = pd.DataFrame(data=d)
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/cif.xlsx")
    writer = pd.ExcelWriter(pathsave, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/cif.csv")
    df.to_csv(pathsave, index=False,encoding='utf-8',sep=';')
    print("writing complete")



def main():
    converter("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.cif")



if __name__ == "__main__":
    main()

    
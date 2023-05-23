import os
import sys
import numpy as np
import pandas  as pd
from prettytable import PrettyTable
from colorama import init, Fore, Back, Style 
#import xlsxWriter 

from LOCSFile import LOCSFile
from BCLFile   import BCLFile
import os.path
import gemmi


def locsreader(f):
    'locs reader'
    if os.path.splitext(f)[1]==".locs":
        file=LOCSFile(f)
        z=file.read_record_locs()
        x,y=next(z)
        print(x)
        print(y)
        return x,y



def readerbcl(f):
    'bclreader'
    if os.path.splitext(f)[1]==".bcl":
        file=BCLFile(f)
        x=file.read_record_bcl()
        base,qual=next(x)
        #print(base)
        print(qual)
        return base,qual

def cifreader(f):
    'cif reader'
    if os.path.splitext(f)[1]==".cif":
        inten=np.random.random(1)
        print(inten)
        return inten




def main():
    'source_dir path sourse'
    #source__dir = "F:/Program Files/Sar/pythondif/s_1_1101.locs"
    #source__dir=os.path.abspath("C:/files/")
    source__dir="C:/Users/evgen/Downloads/DNATOOOLS/files/"
    #print(source__dir)
    #pathsave="F:/Program Files/Sar/pythondif/result.xlsx"
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result.csv")
    #print(os.path.isdir(source__dir))
    if not os.path.isdir(source__dir) == True:
        raise FileNotFoundError('{} does not exists.'.format(source__dir))
    
    i=0

    xcentrall=[]
    ycentrall=[]
    quality=[]
    number=[]
    ainten=[]
    cinten=[]
    ginten=[]
    tinten=[]

    for (dirpath, dirnames, filename) in os.walk(source__dir):
        #print(dirnames)
        #print(filename)
        #print(dirpath)
        for i,filename in enumerate(sorted(filename)):
            filename=os.path.join(dirpath, filename)
            #print(filename)
            if filename.endswith(".locs"):
                name,extension=os.path.splitext(filename)
                xcentr,ycentr=locsreader(filename)
                print()
                print(xcentr)
                print(ycentr)
                print()
                xcentrall.append(np.asarray(xcentr))
                ycentrall.append(np.asarray(ycentr))
            
            if filename.endswith(".bcl"):
                base,qual=readerbcl(filename)
                quality.append(qual)
           
            number.append(i)
            if filename.split("_")[0]=='a' and  filename.endswith(".cif"):
                i=+1
                inten=cifreader(filename)
                ainten.append(inten.sum())
            if filename.split("_")[0]=='g' and  filename.endswith(".cif"):
                inten=cifreader(filename)
                ginten.append(inten.sum())
            if filename.split("_")[0]=='t' and  filename.endswith(".cif"):
                inten=cifreader(filename)
                tinten.append(inten.sum())
            if filename.split("_")[0]=='s' and  filename.endswith(".cif"):
                inten=cifreader(filename)
                cinten.append(inten.sum())
    '''''            
    x=PrettyTable()
    x.add_column('number',number)
    x.add_column('xcentr',xcentrall)
    x.add_column('ycentr',ycentrall)
    x.add_column('a',ainten)
    x.add_column('g',ginten)
    x.add_column('ы',cinten)
    x.add_column('t',tinten)
    x.add_column('quality',quality)
    x.align = "c"
    print(x)
    
                
    data={'number':number,'xcentr':xcentrall,'ycentr':ycentr,'a':ainten,'g':ginten,'c':cinten,'t':tinten,'quality':quality}
    df = pd.DataFrame(data=data)
    print(df)
    '''

    #writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')
    #df.to_excel(writer, sheet_name='Sheet1')
    #writer.close()

    #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
    #df.to_csv(pathsave, index=False ,encoding='utf-8')
    print("writing complete")
    
if __name__ == "__main__":
    main()
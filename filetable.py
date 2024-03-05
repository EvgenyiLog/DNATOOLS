import os
import sys
import numpy as np
import pandas  as pd
from prettytable import PrettyTable
from colorama import init, Fore, Back, Style 
#import xlsxWriter 
import cv2

from LOCSFile import LOCSFile
from BCLFile   import BCLFile
import os.path
import gemmi
from cifreaderq import cifreader
#import picard



def locsreader(f):
    'locs reader'
    if os.path.splitext(f)[1]==".locs":
        file=LOCSFile(f)
        z=file.read_record_locs()
        x,y=next(z)
        #print(x)
        #print(y)
        image=np.zeros((2866, 2944,3))
        image=np.asarray(image,dtype=np.uint8)
        k=len(x)
        for i in range(k):
            #print(x[i])
            image1=cv2.circle(image, (int(x[i]), int(y[i])), 1, (255, 255, 255), -1)
        cv2.imwrite("C:/Users/evgen/Downloads/DNATOOOLS/result/result_bcl.jpg",image1)

        

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





import warnings
import csv
def main():
    warnings.simplefilter("ignore")
    warnings.filterwarnings("default", category=FutureWarning)
    'source_dir path sourse'
    #source__dir = "F:/Program Files/Sar/pythondif/s_1_1101.locs"
    #source__dir=os.path.abspath("C:/files/")
    source__dir="C:/Users/evgen/Downloads/DNATOOOLS/files/"
    #print(source__dir)
    #pathsave="F:/Program Files/Sar/pythondif/result.xlsx"
    quality=[]
    number=[]
    ainten=[]
    cinten=[]
    ginten=[]
    tinten=[]
    quantity=[]
    basef=[]
    ycentrall=[]
    xcentrall=[]


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
                #print(max(xcentr))
                #print(min(xcentr))
                #print(max(ycentr))
                #(min(ycentr))
                print()
                xcentrall.append(np.asarray(xcentr,dtype=float))
                ycentrall.append(np.asarray(ycentr,dtype=float))
                xcentr=np.asarray(xcentr)
                ycentr=np.asarray(ycentr)
                #print(xcentr.shape)
                #print(ycentr.shape)
                quantity.append(np.asarray(xcentr.shape[0],dtype=int))
            
            if filename.endswith(".bcl"):
                base,qual=readerbcl(filename)
                quality.append(np.asarray(qual,dtype=float))
                basef.append(np.asarray(base))
           
            
            if filename.endswith(".cif"):
                intenA,intenC,intenT,intenG,k=cifreader(filename)
                #print(intenA.shape)
                ainten.append(np.asarray(intenA,dtype=float))
                cinten.append(np.asarray(intenC,dtype=float))
                tinten.append(np.asarray(intenT,dtype=float))
                ginten.append(np.asarray(intenG,dtype=float))
                
                
    number=np.arange(1,len(xcentrall)+1)
    #print(number)
    xcentrallnew=[]
    ycentrallnew=[]
    aintennew=[]
    cintennew=[]
    gintennew=[]
    tintennew=[]
    for count, value in enumerate(xcentrall):
        xcentrallnew.append(value)    

    for count, value in enumerate(ycentrall):
        ycentrallnew.append(value) 

    for count, value in enumerate(ainten):
        aintennew.append(value)    

    for count, value in enumerate(cinten):
        cintennew.append(value)  

    for count, value in enumerate(tinten):
        tintennew.append(value)    

    for count, value in enumerate(ginten):
        gintennew.append(value)  
    x=PrettyTable()
    x.add_column('number',number)
    x.add_column('xcentr',xcentrallnew)
    x.add_column('ycentr',ycentrallnew)
    x.add_column('A',aintennew)
    x.add_column('G',gintennew)
    x.add_column('C',cintennew)
    x.add_column('T',tintennew)
    #x.add_column('xcentr',xcentrall)
    #x.add_column('ycentr',ycentrall)
    #x.add_column('A',ainten)
    #x.add_column('G',ginten)
    #x.add_column('C',cinten)
    #x.add_column('T',tinten)
    x.add_column('quantity',quantity)
    x.add_column('quality',quality)
    x.add_column('base',basef)
    x.align = "c"
    print(x)
    
                
    
    #data={'number':number,'xcentr':xcentrall,'ycentr':ycentrall,'quality':quality,'quantity':quantity,'base':basef,'A':ainten,'T':tinten,'C':cinten,'G':ginten}
   
    data={'number':number,'xcentr':xcentrallnew,'ycentr':ycentrallnew,'quality':quality,'quantity':quantity,'base':basef,'A':aintennew,'C':cintennew,'G':gintennew,'T':tintennew}
   

    df = pd.DataFrame(data=data)
    #df=df.pivot(columns=['number','xcentr','ycentr','quality','quantity','base','A','T','C','G'])
    print(df)
    
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/result.xlsx")
    writer = pd.ExcelWriter(pathsave, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    #df.to_excel(pathsave, index=False ,encoding='utf-8', engine='xlsxwriter')
    pathsave=os.path.abspath("C:/Users/evgen/Downloads/DNATOOOLS/result/result.csv")
    df.to_csv(pathsave, index=False,encoding='utf-8',sep=';')
    print("writing complete")
    
if __name__ == "__main__":
    main()
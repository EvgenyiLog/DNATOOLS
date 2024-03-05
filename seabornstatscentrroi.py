<<<<<<< HEAD
import os
import sys
import numpy as np
import pandas  as pd
from prettytable import PrettyTable
from colorama import init, Fore, Back, Style 
#import xlsxWriter 
import matplotlib.pyplot as plt


def xlsxreader(f):
    if os.path.splitext(f)[1]==".xlsx":
        data=pd.read_excel(f)
        print(data)
       
        
        return data
    
import seaborn as sns        
def main():
    data=xlsxreader("C:/Users/evgen/Documents/GitHub/DNATOOLS/result/locs.xlsx")
    plt.figure(figsize=(15,7))
    data.plot()
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.figure(figsize=(15,7))
    
    sns.jointplot(data, x="xcentr", y="ycentr")
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.xlabel('xcentr',fontsize=20)
    plt.ylabel('ycentr',fontsize=20)

    

    plt.figure(figsize=(15,7))
    sns.pairplot(data,x_vars="xcentr", y_vars="ycentr")
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #plt.xlabel('xcentr',fontsize=20)
    #plt.ylabel('ycentr',fontsize=20)

    plt.figure()
    sns.relplot(data,x="xcentr", y="ycentr")
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #plt.xlabel('xcentr',fontsize=20)
    #plt.ylabel('ycentr',fontsize=20)
    '''''

    plt.figure(figsize=(15,7))
    g = sns.PairGrid(data, corner=True)
    g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
    g.map_lower(sns.scatterplot, marker="+")
    g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
    g.add_legend(frameon=True)
    g.legend.set_bbox_to_anchor((.61, .6))
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #plt.xlabel('xcentr',fontsize=20)
    #plt.ylabel('ycentr',fontsize=20)
    '''


    
    plt.show()

    
if __name__ == "__main__":
=======
import os
import sys
import numpy as np
import pandas  as pd
from prettytable import PrettyTable
from colorama import init, Fore, Back, Style 
#import xlsxWriter 
import matplotlib.pyplot as plt


def xlsxreader(f):
    if os.path.splitext(f)[1]==".xlsx":
        data=pd.read_excel(f)
        print(data)
       
        
        return data
    
import seaborn as sns        
def main():
    data=xlsxreader("C:/Users/evgen/Documents/GitHub/DNATOOLS/result/locs.xlsx")
    plt.figure(figsize=(15,7))
    data.plot()
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.figure(figsize=(15,7))
    
    sns.jointplot(data, x="xcentr", y="ycentr")
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    plt.xlabel('xcentr',fontsize=20)
    plt.ylabel('ycentr',fontsize=20)

    

    plt.figure(figsize=(15,7))
    sns.pairplot(data,x_vars="xcentr", y_vars="ycentr")
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #plt.xlabel('xcentr',fontsize=20)
    #plt.ylabel('ycentr',fontsize=20)

    plt.figure()
    sns.relplot(data,x="xcentr", y="ycentr")
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #plt.xlabel('xcentr',fontsize=20)
    #plt.ylabel('ycentr',fontsize=20)
    '''''

    plt.figure(figsize=(15,7))
    g = sns.PairGrid(data, corner=True)
    g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
    g.map_lower(sns.scatterplot, marker="+")
    g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
    g.add_legend(frameon=True)
    g.legend.set_bbox_to_anchor((.61, .6))
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    #plt.xlabel('xcentr',fontsize=20)
    #plt.ylabel('ycentr',fontsize=20)
    '''


    
    plt.show()

    
if __name__ == "__main__":
>>>>>>> ea6a11630c874ef013efd635151588613733c77e
    main()
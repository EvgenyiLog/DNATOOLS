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
from sklearn.decomposition import FastICA,PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
import dtaidistance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import Lasso,Ridge
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from scipy.stats import f_oneway
from dtaidistance import dtw, clustering
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



 




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
            
           
            
            if filename.endswith(".cif"):
                intenA,intenC,intenT,intenG,k=cifreader(filename)
                #print(intenA.shape)
                ainten.append(np.asarray(intenA,dtype=float))
                cinten.append(np.asarray(intenC,dtype=float))
                tinten.append(np.asarray(intenT,dtype=float))
                ginten.append(np.asarray(intenG,dtype=float))


    number=np.arange(1,len(ginten)+1)
    #print(number)
              
    x=PrettyTable()
    x.add_column('number',number)
    x.add_column('A',ainten)
    x.add_column('G',ginten)
    x.add_column('C',cinten)
    x.add_column('T',tinten)
    x.align = "c"
    print(x)

    

    
    x=np.column_stack((ainten,ginten,cinten,tinten))
    print(x.shape)
    df = pd.DataFrame(x,
                  columns=['intensitivityA','intensitivityG','intensitivityC','intensitivityT'])
    print(df.keys())
    le = preprocessing.LabelEncoder()
    le.fit(df.keys())

    
    print(list(le.classes_))
    y=le.transform(le.classes_)

    print(y)
    clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft')
    #print(df.values)
    
    #eclf = eclf.fit(x, y)
    kmeans = KMeans(n_clusters=4, random_state=None, n_init="auto").fit(df.values)
    print(kmeans.labels_)
    X, y = make_classification(n_features=4)
    clf = ExtraTreesClassifier(n_estimators=200)
    clf.fit(X, y)
    clf.predict(df.values)
    print(clf.classes_)
    #new_center = dtaidistance.dtw_barycenter.dba(df.values, center, use_c=True)
    #new_center = dtaidistance.dtw_barycenter.dba_loop(df.values, center, max_it=10, thr=0.0001, use_c=True)
    model = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=3)
    cluster_idx = model.fit(df.values)
    print(cluster_idx)
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(X, y)
    print(clf.classes_)
    clf = BaggingClassifier(estimator=SVC(),n_estimators=10).fit(X, y)
    print(clf.predict(x))
    
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    print(clf.predict(x))
    clf = LogisticRegression(random_state=0).fit(X, y)
    print(clf.predict(x))
    clf=SVC()
    clf.fit(X, y)
    print(clf.predict(x))
    x=np.c_[ainten,ginten,cinten,tinten]
    pca = PCA(n_components=4)
    
    H = pca.fit_transform(x)
    ica = FastICA(n_components=4)
    S_ = ica.fit_transform(x)  # Get the estimated sources
    A_ = ica.mixing_  # Get estimated mixing matrix

    f,p=f_oneway(df['intensitivityA'].values,df['intensitivityG'].values)
    print(f)
    print(p)
    f,p=f_oneway(df['intensitivityC'].values,df['intensitivityT'].values)
    print(f)
    print(p)



    boxplot = df.boxplot(column=['intensitivityA','intensitivityG','intensitivityC','intensitivityT'])
    plt.figure('Violinplot',figsize=(15,7))   
    sns.violinplot(data=df)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет 
    

    plt.figure('Boxplot',figsize=(15,7))    
    sns.boxplot(df)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
    

    #plt.figure('Histplot',figsize=(15,7))  
    #sns.histplot(df)
    #plt.grid(True)
    #plt.tick_params(labelsize =20,#  Размер подписи
                    #color = 'k')   #  Цвет делений
    plt.figure('Barplot',figsize=(15,7))  
    sns.barplot(df)
    plt.grid(True)
    plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        
    plt.show()




if __name__ == "__main__":
    main()
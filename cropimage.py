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
    


def cropimage(filename1,filename2,width, height, angle):
    xcentr,ycentr=locsreader(filename1)
    image=tiffreader(filename2)
    xcentr=np.asarray(xcentr,dtype=float)
    ycentr=np.asarray(ycentr,dtype=float)
    #print(xcentr.shape)
    mask=np.zeros_like(image,dtype=np.uint8)
    k=20
    plt.figure('Crop rect',figsize=(15,7))
    for i in range(k):
        plt.subplot(k,1,1+i)
        #print(i)
        #print(xcentr[i])
        #print()
        #print(ycentr[i])
        if ycentr[i]>10:
            rect=((xcentr[i],ycentr[i]),(width, height), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(box)
            #print(box[1][0])
            imagecrop=image[box[1][0]:box[2][0],box[2][1]:box[3][1]]
            filename=i
            save_dir="C:/Users/evgen/Downloads/DNATOOOLS/result/"
            filepath=os.path.join(save_dir, str(filename))
            filepath=os.path.splitext(os.path.abspath(filepath))[0]+".jpg"
            print(filepath)
            cv2.imwrite(filepath,imagecrop)
        else:
            rect=((xcentr[i],ycentr[i]),(width, height), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #print(box)
            #print(box[1][0])
            imagecrop=image[box[1][0]:box[2][0],0:10]
            filename=i
            save_dir="C:/Users/evgen/Downloads/DNATOOOLS/result/"
            filepath=os.path.join(save_dir, str(filename))
            filepath=os.path.splitext(os.path.abspath(filepath))[0]+".jpg"
            print(filepath)
            cv2.imwrite(filepath,imagecrop)
            

        #print(imagecrop.shape)
        plt.imshow(imagecrop,cmap='gray',vmin=imagecrop.min(),vmax=imagecrop.max())
        plt.tick_params(labelsize =20,#  Размер подписи
                    color = 'k')   #  Цвет делений
        


    
        

    


    plt.show()




def main():
    cropimage("C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101.locs","C:/Users/evgen/Downloads/DNATOOOLS/files/s_1_1101_c.tif",5, 5, 0)



if __name__ == '__main__':
    main()

    
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:49:43 2018

@author: Pooja
"""
#http://answers.opencv.org/question/94448/module-cv2-has-no-attribute-createbackgroundsubtractormog/
import os
import numpy as np
import cv2
import pandas as pd
from IPython.display import Image #images from file
from matplotlib import pyplot as plt #cv2 images
from scipy.io import loadmat
from scipy.misc import imsave
from imageio import imwrite
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\44Col\\0_0')
img = os.listdir()
print(len(img))
for i in range(len(img)):
    frame = cv2.imread(img[i], 0)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dim = (32, 32)
    imf = np.float32(frame)/255.0  # float conversion/scale
    dst = cv2.dct(imf) 
    print(dst[10:15,10:15])          # the dct
    #frame = np.uint8(dst)*255.0    # convert back
    
    cv2.namedWindow(str(i))
    cv2.imshow(str(i), frame)
    k=cv2.waitKey(0)
    #a
    if k==97:
        i+=1
    #s    
    elif k==115.:
        i-=1
        if i==-1:
            i=0
    elif k==27: #ESC
        break
cv2.destroyAllWindows() 
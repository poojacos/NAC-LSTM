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
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\frm\\youtube_vmfwbST5QD4\\')
img = os.listdir()
print(len(img))
#MOG working good
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg = cv2.createBackgroundSubtractorGMG()
for i in range(len(img)):
    frame = cv2.imread(img[i])
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dim = (32, 32)
    #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
    fgmask = fgbg.apply(frame)  
    #morphological transform
    kernel = np.ones((3, 3),np.uint8)
    #fgmask = cv2.dilate(fgmask.copy(),kernel,iterations = 1)
    fgmask = cv2.morphologyEx(fgmask.copy(), cv2.MORPH_GRADIENT, kernel)
#    kernel = np.ones((2,2),np.uint8)
#    fgmask = cv2.dilate(fgmask.copy(),kernel,iterations = 1)
    cv2.namedWindow(str(i))
    cv2.imshow(str(i), fgmask)
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
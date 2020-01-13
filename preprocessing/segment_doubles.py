# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:29:15 2018

@author: Pooja
"""

#Bounding Boxes and Segmented Images
import os
import numpy as np
import cv2
import pandas as pd
from IPython.display import Image #images from file
from matplotlib import pyplot as plt #cv2 images
from scipy.io import loadmat
from scipy.misc import imsave
from imageio import imwrite
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\')

path_track = 'figure_tracks\\youtube_RZ2k-tsX5KE\\'
path_img = 'frm\\youtube_RZ2k-tsX5KE\\'

img_list = os.listdir(path_img)

#img_seg =loadmat(path_seg + seg_list[0])
#course = img_seg['fg_mask_coarse']
#fine = img_seg['fg_mask_fine']
#type(course) ---> ndarray
shp = list()

#function to make bounding boxes of normal images
def bb_normal():
    folder = 'doubles'
    #p1, p2 if there are two players, p0 if there is only one image
    it = len(img_list)
    for i in range(1,it):
        img = cv2.imread(path_img + img_list[i])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        track = loadmat(path_track + 'tracked_boxes.mat')
        track = track['all_tracks']
        #us-frame-player-us-item-us-val
        print(i)
        #creating folder for each bounding box for a particular image
#        if not os.path.exists(folder+'\\'+str(i)):
#            os.mkdir(folder+'\\'+str(i))
#            os.chdir(folder+'\\'+str(i)+'\\')
#        else:
#            os.chdir(folder+'\\'+str(i)+'\\')
            
        loops = track[0][i].shape[0]
        for j in range(loops):
            t0 = track[0][i][j][0][2][0]
#            if int(t0[2])<25:
#                continue
            img1 = img[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])]
            shp.append(img1.shape)
            print(str(i) + '_' + str(j) + '.jpg')
            imwrite(folder + '\\' + str(i) + '_' + str(j) + '.jpg', img1)
  
bb_normal()     
#i=900
#while(True):
#    #ll = os.listdir('BB\\')
#    ll = img_list
#    img = cv2.imread(path_img + ll[i])
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    #ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#    #blur = cv2.GaussianBlur(img,(5,5),0)
#    #ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#    
#    cv2.namedWindow(str(i))
#    cv2.imshow(str(i), img)
#    k=cv2.waitKey(0)
#    if k==97:
#        i+=1
#    elif k==115.:
#        i-=1
#        if i==-1:
#            i=0
#    elif k==27: #ESC
#        break
#cv2.destroyAllWindows()    
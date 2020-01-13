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
from matplotlib import pyplot as plt #cv2 images
from scipy.io import loadmat
from scipy.misc import imsave
from imageio import imwrite
from PIL import Image
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\')
#Used this for the singles matches non pre processed images
#path_seg = 'fgmask\\youtube_vmfwbST5QD4\\' 
#path_track = 'figure_tracks\\youtube_vmfwbST5QD4\\'
#path_img = 'frm\\youtube_vmfwbST5QD4\\'

#Used this for rough rule base concept median images
path_uiuc = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\'
path_seg = 'fgmask\\youtube_vmfwbST5QD4\\' 
path_track = 'figure_tracks\\youtube_vmfwbST5QD4\\'
path_img = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\rough_image\\'

seg_list = os.listdir(path_seg)
img_list = os.listdir(path_img)

#img_seg =loadmat(path_seg + seg_list[0])
#course = img_seg['fg_mask_coarse']
#fine = img_seg['fg_mask_fine']
#type(course) ---> ndarray
shp = list()

def create_segimg():
    it = len(seg_list)
    for i in range(it):
        img_seg =loadmat(path_seg + seg_list[i])
        course = img_seg['fg_mask_coarse']
        fine = img_seg['fg_mask_fine']
        course *= 255
        fine *= 255
        imwrite('cseg\\' + str(i) + '.jpg', course)
        imwrite('fseg\\' +str(i) + '.jpg', fine)

#function to make bounding boxed images of the segmented images.        
def bb_make():
    it = len(seg_list)
    for i in range(it):
        course = cv2.imread('cseg\\' + str(i) + '.jpg')
        fine = cv2.imread('fseg\\' +str(i) + '.jpg')
        track = loadmat(path_track + 'tracked_boxes.mat')
        track = track['all_tracks']
        #us-frame-player-us-item-us-val
        print(i)
        if track[0][i].shape[0] >= 2:
            #player1 bb
            t0 = track[0][i][0][0][2][0]
            #player2 bb
            t1 = track[0][i][1][0][2][0]
            
            imwrite('cBB\\p1_' + str(i) + '.jpg', course[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])])
            imwrite('cBB\\p2_' +str(i) + '.jpg', course[int(t1[1]):int(t1[1]+t1[3]), int(t1[0]):int(t1[0]+t1[2])])
            
            imwrite('fBB\\p1_' + str(i) + '.jpg', fine[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])])
            imwrite('fBB\\p2_' +str(i) + '.jpg', fine[int(t1[1]):int(t1[1]+t1[3]), int(t1[0]):int(t1[0]+t1[2])])
        else:
            t0 = track[0][i][0][0][2][0]
            imwrite('cBB\\p0_' + str(i) + '.jpg', course[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])])
            imwrite('fBB\\p0_' +str(i) + '.jpg', fine[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])])

#function to make bounding boxes of normal images
def bb_normal():
    os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\')
    #folder = 'BBuncut'
    folder = 'Rough'
    #p1, p2 if there are two players, p0 if there is only one image
    it = len(seg_list)
    for i in range(1,it):
        #for singles match
        #img = cv2.imread(path_img + img_list[i])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.open(path_img + img_list[i])
        img = np.array(img)
        
        track = loadmat(path_track + 'tracked_boxes.mat')
        track = track['all_tracks']
        #us-frame-player-us-item-us-val
        print(i)
        if track[0][i].shape[0] == 2:
            t0 = track[0][i][0][0][2][0]
            t1 = track[0][i][1][0][2][0]
            img1 = img[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])]
            img2 = img[int(t1[1]):int(t1[1]+t1[3]), int(t1[0]):int(t1[0]+t1[2])]
            shp.append(img1.shape)
            shp.append(img2.shape)
            if t0[1] < t1[1]:
                imwrite(folder + '\\pt_' + str(i) + '.jpg', img1)
                imwrite(folder + '\\pb_' +str(i) + '.jpg', img2)
            else:    
                imwrite(folder + '\\pt_' + str(i) + '.jpg', img2)
                imwrite(folder + '\\pb_' +str(i) + '.jpg', img1)
        else:
            t0 = track[0][i][0][0][2][0]
            img1 = img[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])]
            shp.append(img1.shape)
            imwrite(folder + '\\p0_' + str(i) + '.jpg', img1)
            
#function to bound the text files  
def bb_text():
    os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\')
#    txt = 'rough_text\\'
#    txt_seg = 'rough_text_seg\\'
    txt_seg_blur = 'rough_text_seg_blur\\'
    #change the values of folders only once, here
    folder = txt_seg_blur
    os.chdir(folder)
    print(os.getcwd())
    it = os.listdir()    
    it_len = len(it)
    print("Number of files = %d"%it_len)
    
    #create folder to save
    folder = folder[:-1]+'_cut'
    print('saving folder = {}'.format(folder))
    
    for i in range(it_len):       
        #reading the stored text file instead of image
        img = np.loadtxt(it[i],dtype=float)
        track = loadmat(path_uiuc + path_track + 'tracked_boxes.mat')
        track = track['all_tracks']
       
        if track[0][i].shape[0] == 2:
            t0 = track[0][i][0][0][2][0]
            t1 = track[0][i][1][0][2][0]
            img1 = img[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])]
            img2 = img[int(t1[1]):int(t1[1]+t1[3]), int(t1[0]):int(t1[0]+t1[2])]

            if t0[1] < t1[1]:
                np.savetxt(path_uiuc + folder + '\\pt_' + str(i) + '.txt', img1, fmt='%.5f')
                np.savetxt(path_uiuc + folder + '\\pb_' +str(i) + '.txt', img2, fmt='%.5f')
            else:    
                np.savetxt(path_uiuc + folder + '\\pt_' + str(i) + '.txt', img2, fmt='%.5f')
                np.savetxt(path_uiuc + folder + '\\pb_' +str(i) + '.txt', img1, fmt='%.5f')
        else:
            t0 = track[0][i][0][0][2][0]
            img1 = img[int(t0[1]):int(t0[1]+t0[3]), int(t0[0]):int(t0[0]+t0[2])]
            np.savetxt(path_uiuc + folder + '\\p0_' + str(i) + '.txt', img1, fmt='%.5f')
    print('all txt files cut and saved')  

bb_text()     
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
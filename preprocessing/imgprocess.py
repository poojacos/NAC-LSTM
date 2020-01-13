# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:49:43 2018

@author: Pooja
"""

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

path_img = 'BB\\'
path_gray = 'gray\\'
path_clips = 'clips\\'
img_list = os.listdir(path_img)
gray_list = list()
shp = list()

def store_shapes():
    it = len(img_list)
    for i in range(it):
        print(i)
        im = cv2.imread(path_img + img_list[i],0)
        shp.append(im.shape)
"""        
store_shapes()  
shp = np.asarray(shp)
#max-- ht = 74 wd = 65
#min-- ht = 18 wd = 12
#avg-- ht = 42 wd = 28
ht,wd = np.max(shp, axis = 0)
        
"""  
def resizeimg():      
    folder1 = 'gray\\'
    dim = (40, 30)
    it = len(img_list)
    for i in range(it):
        im = cv2.imread(path_img + img_list[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, dim, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(folder1 + img_list[i], im)  
        
def clips_save(path1, start, end, ext = '.jpg'):   
    for j in range(int(start), int(end+1)): 
        #incorrect images not read
        avoid = ['pt_105.jpg', 'pt_106.jpg', 'pt_623.jpg', 'pt_624.jpg', 'pt_625.jpg', 'pt_626.jpg', 'pt_627.jpg','pt_628.jpg']
        if j == 1546 or j == 1623 or j == 3070:
            continue
        
        #creating image name
        val = player + '_' + str(j) + ext 
        if val in avoid:
            continue
        
        img = cv2.imread(path1 + path_gray + val)
        ret = cv2.imwrite(val, img)
        if ret == False:
            print(ret, val)
    return True
    
gray_list = os.listdir(path_gray)
path1 = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\'
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\Project\\') 
file1 = pd.read_csv('label_data.csv', header = 0)
file2 = pd.read_csv('Book1_today.csv', header = 0)
file3 = pd.concat([file2,file1], axis = 0)
print(file3.head())
it = file3.shape[0]
ext = '.jpg'
r1, se, sm, b1, l1, f1 = 0,0,0,0,0,0
for i in range(it):
    player = file3['player'].iloc[i]
    player = player.strip()
    start = file3['start'].iloc[i]
    end = file3['end'].iloc[i]
    no_play = file3['no play'].iloc[i]
    #plays
    react = file3['react'].iloc[i]
    serve = file3['serve'].iloc[i]
    smash = file3['smash'].iloc[i]
    backhand = file3['backhand'].iloc[i]
    lob = file3['lob'].iloc[i]
    forehand = file3['forehand'].iloc[i]

    if no_play == 1:
        continue    
#    #checking for error in data entry - No error found
#    test = np.asarray(file3[['react','serve','smash','backhand', 'lob','forehand']].iloc[i])
#    if np.sum(test) != 1:
#        print('error')
#        print(i)
    #BB has images from value 1 to 3690
    if react == 1:
        path2 = path1 + path_clips + 'react'
        if not os.path.exists(path2):
            os.makedirs(path2)
        os.chdir(path2)
        if not os.path.exists(str(r1)):    
            os.makedirs(str(r1)) 
        os.chdir(str(r1))
        ret = clips_save(path1, start, end)
        r1 += 1
        
    elif serve == 1:
        path2 = path1 + path_clips + 'serve'
        if not os.path.exists(path2):
            os.makedirs(path2)
        os.chdir(path2)
        if not os.path.exists(str(se)):    
            os.makedirs(str(se)) 
        os.chdir(str(se))
        ret = clips_save(path1, start, end)
        se += 1
       
    elif smash == 1:
        path2 = path1 + path_clips + 'smash'
        if not os.path.exists(path2):
            os.makedirs(path2)
        os.chdir(path2)
        if not os.path.exists(str(sm)):    
            os.makedirs(str(sm)) 
        os.chdir(str(sm))
        ret = clips_save(path1, start, end)
        sm += 1    
    
    elif backhand == 1:
        path2 = path1 + path_clips + 'backhand'
        if not os.path.exists(path2):
            os.makedirs(path2)
        os.chdir(path2)
        if not os.path.exists(str(b1)):    
            os.makedirs(str(b1)) 
        os.chdir(str(b1))
        ret = clips_save(path1, start, end)
        b1 += 1
        
    elif lob == 1:
        path2 = path1 + path_clips + 'lob'
        if not os.path.exists(path2):
            os.makedirs(path2)
        os.chdir(path2)
        if not os.path.exists(str(l1)):    
            os.makedirs(str(l1))
        os.chdir(str(l1))
        ret = clips_save(path1, start, end)
        l1 += 1
        
    elif forehand == 1:
        path2 = path1 + path_clips + 'forehand'
        if not os.path.exists(path2):
            os.makedirs(path2)
        os.chdir(path2)
        if not os.path.exists(str(f1)):    
            os.makedirs(str(f1)) 
        os.chdir(str(f1))
        ret = clips_save(path1, start, end)
        f1 += 1    
        
    else:
        print('Error!')
        
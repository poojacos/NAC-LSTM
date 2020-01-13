# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:56:11 2018

@author: Pooja
"""
import numpy as np
import os
import cv2
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Permute, Reshape, Lambda, RepeatVector, Input, Bidirectional
#from keras.layers import Merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed	
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
MaxPooling2D)
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint	

from keras.layers import multiply
from keras.layers.core import *
from keras.models import *

#from attention_utils import get_activations, get_data_recurrent

os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\clips\\')
path_seg = 'fgmask\\youtube_vmfwbST5QD4\\' 
path_track = 'figure_tracks\\youtube_vmfwbST5QD4\\'
path_img = 'frm\\youtube_vmfwbST5QD4\\'

max_seq = 44

def expand(imglist, vidlen, max_seq, frames):
    #check for indexing in img list and start end values
    diff = max_seq - vidlen
    if diff >= vidlen:
        q = int(diff/vidlen)
        for i in range(vidlen):
            temp = q
            while(temp>-1 and len(frames)<max_seq):
                frames.append(imglist[i])
                temp -= 1
                
        if len(frames) == max_seq:
            frames = np.asarray(frames)
            return frames
            
        elif len(frames) < max_seq and 2*len(frames) > max_seq:
            vidlen = len(frames)
            frames = expand(frames, vidlen, max_seq, frames)
            return frames
            
        else:
            print('Error!')
            
    else:
        #if it doesn't comes from previous part
        if len(frames)==0:
            for i in range(vidlen):  
                frames.append(imglist[i])
        #logic 
        idx = np.round(np.linspace(0, len(frames) - 1, diff)).astype(int)
        it = idx.shape[0]
        i, shift = 0, 0
        frames = np.asarray(frames)
        while(i<it and len(frames) < max_seq):
             cpy = frames[idx[i] + shift]
             frames = np.insert(frames, idx[i] + shift + 1, cpy)
             shift += 1
             i += 1
       
        if len(frames) == max_seq:
            return frames
        
        else:
            print('Error!')
    
    print(len(frames),diff)  
    frames = np.asarray(frames)      
    return frames        
            
    
def expand_seq(max_seq):
    videos =list()
    x = list()
    y = list()
    action = os.listdir()
    for i1 in range(len(action)):
        num = os.listdir(action[i1] + '\\')
        for i2 in range(len(num)):
            frames = list()
            imglist = list()
            names = os.listdir(action[i1] + '\\' + num[i2] + '\\')
            print(action[i1] + '\\' + num[i2] + '\\')
            for i3 in range(len(names)):
                if len(frames)<max_seq:
                    frames.append(names[i3])

            imglist = expand(frames, len(frames), max_seq, imglist) 
            
            #creating dataset
            temp = list()
            for i3 in range(len(imglist)):
                img = cv2.imread(action[i1] + '\\' + num[i2] + '\\' + imglist[i3])
                temp.append(img)
            temp = np.asarray(temp)
            x.append(temp)
            if action[i1] == 'backhand':
               y.append(0) 
            elif action[i1] == 'forehand':
               y.append(1) 
            elif action[i1] == 'lob':
               y.append(2)
            elif action[i1] == 'react':
               y.append(3) 
            elif action[i1] == 'serve':   
               y.append(4) 
            elif action[i1] == 'smash':
               y.append(5)
            else:
                print('Error!')
               
            videos.append(imglist)   
    return videos, x, y  

videos, x, y = expand_seq(max_seq)

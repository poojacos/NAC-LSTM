# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:56:11 2018

@author: Pooja
"""
import numpy as np
import os
import cv2


#from attention_utils import get_activations, get_data_recurrent

os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\clipsCol\\')
path_seg = 'fgmask\\youtube_vmfwbST5QD4\\' 
path_track = 'figure_tracks\\youtube_vmfwbST5QD4\\'
path_img = 'frm\\youtube_vmfwbST5QD4\\'
path_44 = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\44Col\\'
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
            
#for sequences saved action wise in folders    
def expand_seq(max_seq):
    vid44 = 0
    videos =list()
    x = list()
    y = list()
    action = os.listdir()
    for i1 in range(len(action)):
        os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\clipsCol\\')
        num = os.listdir(action[i1] + '\\')
        for i2 in range(len(num)):
            os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\clipsCol\\')
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
            
            path_save = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\44Col\\' + str(vid44) + '_' + str(y[vid44])
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            os.chdir(path_save)
            for i in range(temp.shape[0]):
                cv2.imwrite(str(i)+'.jpg', temp[i])
            vid44 += 1
            
            videos.append(imglist)   
    return videos, x, y  

videos, x, y = expand_seq(max_seq)
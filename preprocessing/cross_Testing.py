# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:29:14 2018

@author: Pooja
"""

#testing model of doubles on singles and vice versa
#import tensorflow as tf
#import keras
import os
import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import confusion_matrix
path_uiuc = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\'
path_44 = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\44D1\\'

mod = load_model(path_uiuc + 'tcn_models\\tcn_s_64_adam_1.h5')
#mod.predict()

x = list()
y = list()
#reading color image - changin to gray and subtracting mean
vidlist = os.listdir(path_44)
it = len(vidlist)
for i in range(it):
    s1 = vidlist[i]
    _, typ = s1.split('_')

    if typ == '3':
        y.append(3)
    
    elif typ == '0':
       y.append(0)
       
    elif typ == '1':
       y.append(1)
       
    elif typ == '2':
       y.append(2)
       
    elif typ == '4':  
        continue
       #y.append(4)
       
    elif typ == '5':
       y.append(5)
    else:
        print('Error!' + str(typ))
        
    os.chdir(path_44 + s1)   
    img_list = os.listdir()
    it0 = len(img_list)
    temp = list()
    dim = (32,32)
    for j in range(it0):
        img = cv2.imread(img_list[j])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        mn1 = np.mean(img)
        st1 = np.std(img)
        img = (img.copy() - mn1) / st1
        img = img.copy().flatten()
        temp.append(img)
        
    temp = np.asarray(temp)
    x.append(temp)
    
X = np.asarray(x)
Y = np.asarray(y)
print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.3, random_state=4, shuffle=True)
y_pred = mod.predict(X_test)
y_pred = y_pred.argmax(axis=-1)
y_test = np.asarray(y_test)
it = y_test.shape[0]
tp, tn, fp, fn = 0,0,0,0
for i in range(it):
    if y_test[i]==y_pred[i]:
        tp+=1
print("{} percentage of correct prediction".format(tp/it))        
print("%d correct out of %d"%(tp,it))    
                                                 
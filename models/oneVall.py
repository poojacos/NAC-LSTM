# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:30:00 2018

@author: Pooja
"""
#NAC using different functions like relu, elu etc.
import numpy as np
import os
import cv2
import keras
from tcn import tcn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Flatten, Dropout, Input, Bidirectional,BatchNormalization, Reshape
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed	
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint	
from nalu import NALU
from nac import NAC
import time
max_seq = 44
num = np.zeros(6)
path_uiuc = 'C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\'
x = list()
y = list()
#reading color image - changin to gray and subtracting mean
data_fold = ['44D1', '44D2', '44Col']
for fold in data_fold:
    path_44 = path_uiuc + fold + '\\'
    print(path_44)
    vidlist = os.listdir(path_44)
    it = len(vidlist)
    for i in range(it):
        s1 = vidlist[i]
        _, typ = s1.split('_')
        #NO REACT
        if typ == '3':
            y.append(0)
        #backhand
        elif typ == '0':
           y.append(0)
        #forehand   
        elif typ == '1':
           y.append(0)
        #lob   
        elif typ == '2':
           y.append(0)
        #serve   
        elif typ == '4': 
           y.append(1)
         #smash  
        elif typ == '5':
           y.append(0)
        else:
            print('Error!' + str(typ))
            
        num[int(typ)] +=1
        
        os.chdir(path_44 + s1)   
        img_list = os.listdir()
        it0 = len(img_list)
        temp = list()
        dim = (32,32)
        for j in range(it0):
            img = cv2.imread(img_list[j])
            #print(str(i)+'_'+str(j))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
            mn1 = np.mean(img)
            st1 = np.std(img)
            img = (img.copy() - mn1) / st1
            img = img.copy().flatten()
            temp.append(img)
        temp = np.asarray(temp)     
        x.append(temp)
        
strokes = {'backhand':num[0], 'forehand':num[1], 'lob':num[2], 'react':num[3], 'serve':num[4], 'smash':num[5]}
for i in strokes.keys():
    print("{} - %d".format(i)%(strokes[i]))
    
x = np.asarray(x)
y = np.asarray(y)
print(x.shape)
x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40,shuffle =True, stratify=y)
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
print(X_test.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#y_train = np_utils.to_categorical(y_train, 6)
y_test = np_utils.to_categorical(y_test, 2)        


seed = 1   
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
ii = 0 

# create model
i = Input(batch_shape=(None, 1024,44))
#o = Dense(44,input_shape=(1024,44))(i)
o = NAC(5, use_gating = True)(i)
o = Flatten()(o)
o = Dense(500)(o)
#o = LSTM(100, return_sequences=False, dropout=0.25)(o)
o = Dense(2, activation='sigmoid')(o)    
model = Model(inputs=[i], outputs=[o])
print(model.summary())
start = time.time()
for train, test in kfold.split(X_train, y_train):
    print('------------ITERATION : ' + str(ii) + '  -------------------') 
	# Compile model
    #sgd = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #sgd = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	# Fit the model
    Yy = np_utils.to_categorical(y_train, 2)
    model.fit(X_train[train], Yy[train], epochs=110, batch_size=32, verbose=0)
    model.save('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\nalu_ov'+str(ii)+'.h5')
    ii += 1
    #print(model.predict(X[train], batch_size=32))
	# evaluate the model
    scores = model.evaluate(X_train[test], Yy[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
exectime = time.time()-start    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

final_scores = []
for i in range(1):
  mod = load_model('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\nalu_ov'+str(i)+'.h5')
  eval1 = time.time()
  scores = model.evaluate(X_test, y_test, verbose=1)
  evaltim = time.time()-eval1
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  final_scores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(final_scores), np.std(final_scores)))
print(exectime, evaltim)

"""
Created on Wed Sep 19 18:56:11 2018

@author: Pooja

"""
#acc: 33.3%


import numpy as np
import os
import cv2
import keras
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Dropout, Permute, Reshape, Lambda, RepeatVector, Input, Bidirectional,BatchNormalization
from keras.layers import merge
#from keras.layers import Merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed	
from keras.layers.convolutional import (Conv2D,MaxPooling2D)
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint	,EarlyStopping

from keras.layers import multiply
from keras.layers.core import *
from keras.models import *

#from attention_utils import get_activations, get_data_recurrent
#
#os.chdir('/user1/temp/cscr/pooja_t/clips/')
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\clips\\')
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
        num = os.listdir(action[i1] + '/')
        for i2 in range(len(num)):
            frames = list()
            imglist = list()
            names = os.listdir(action[i1] + '/' + num[i2] + '/')
            print(action[i1] + '/' + num[i2] + '/')
            for i3 in range(len(names)):
                if len(frames)<max_seq:
                    frames.append(names[i3])

            imglist = expand(frames, len(frames), max_seq, imglist) 
            
            #creating dataset
            temp = list()
            for i3 in range(len(imglist)):
                img = cv2.imread(action[i1] + '/' + num[i2] + '/' + imglist[i3])
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
x = np.asarray(x)
y = np.asarray(y)
#x.shape = (235, 44, 30, 40, 3)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42,shuffle =True)
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, 6)
y_test = np_utils.to_categorical(y_test, 6)
#
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    print(inputs.shape[2])
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(max_seq, activation='softmax')(a)
    if False:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = keras.layers.multiply([inputs, a_probs])
    print(inputs.shape, a_probs.shape)
    return output_attention_mul

#model = Sequential()
x = Input((44,30,40,3))
model = (BatchNormalization())(x)
model = (TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same')))(model)
model = (TimeDistributed(Conv2D(32, (3,3),  activation='relu')))(model)
model = (TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))(model)
model = (Dropout(0.25))(model)

#model = (BatchNormalization())(model)
model = (TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')))(model)
model = (TimeDistributed(Conv2D(64, (3,3),  activation='relu')))(model)
model = (TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))(model)
model = (Dropout(0.25))(model)

#model = (BatchNormalization())(model)
model = (TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same')))(model)
model = (TimeDistributed(Conv2D(128, (3,3),  activation='relu')))(model)
model = (TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))(model)

model = (TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same')))(model)
model = (TimeDistributed(Conv2D(256, (3,3),  activation='relu')))(model)
model = (TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))(model)
model = (Dropout(0.25))(model)

model = (TimeDistributed(Flatten()))(model)
model = (Dropout(0.5))(model)
#AFTER LSTM
model = LSTM(128, return_sequences=True)(model)
model = attention_3d_block(model)
model = Flatten()(model)
model = (Dense(6, activation='softmax'))(model)
model = Model(input = x, output = model)
opt = keras.optimizers.Adam(lr=0.0005)

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
## filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
## checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
## callbacks_list = [checkpoint]
#mm = ModelCheckpoint('/user1/temp/cscr/pooja_t/', monitor='loss', verbose=0, save_best_only=True, mode='auto', period=1)
checkpoint = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto')
callbacks_list = [checkpoint]
model.fit(X_train, y_train, batch_size=32, epochs=70, verbose=1, callbacks=callbacks_list, validation_split=0.1)
#model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)
#model.load_weights('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\B128-256.h5')
score = (model.evaluate(X_test,y_test))
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.save('/user1/temp/cscr/pooja_t/after64.h5')
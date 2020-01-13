# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:25:50 2018

@author: Pooja
"""
#NOT EFFECTIVE ON frm folder or BB
import os
import numpy as np
import cv2
import time
import pandas as pd
from IPython.display import Image #images from file
from matplotlib import pyplot as plt #cv2 images
from scipy.io import loadmat
from scipy.misc import imsave
from imageio import imwrite
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\BB\\')
img = os.listdir()
print(len(img))
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\')
classes = open('categories.txt').read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(classes) - 1, 3),dtype="uint8")
COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

net = cv2.dnn.readNet('model-cityscapes.net')
frame = cv2.imread('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\BB\\' + img[2])
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)
#c = np.asarray(blob)
#cv2.imwrite('c.jpg', c)
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()
print('running time %.2f secs'%(end- start))
(numClasses, height, width) = output.shape[1:4]
classMap = np.argmax(output[0], axis=0)

mask = COLORS[classMap]
mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")
cv2.imwrite('b.jpg',output)
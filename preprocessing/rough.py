# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:41:50 2018

@author: Pooja
"""

"""
can be the part capturing temporal information - providing direction of movement
(left/right problem, is direction of movement so obviosly clear that it's helping?)
i see that when player is moving there are little whites
these images can be fed to an LSTM
"""
import os
import numpy as np
import cv2
#os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\frm\\youtube_vmfwbST5QD4\\')
os.chdir('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\fseg\\')
a = os.listdir()

for i in range(5,len(a)):
    print(a[i])
    img = cv2.imread(a[i],0)
    mn1 = np.mean(img)
    st1 = np.std(img)
    img = (img.copy() - mn1) / st1
    temp = list()
    for j in range(5):
        i1 = cv2.imread(a[i-j],0)
        mn2 = np.mean(i1)
        st2 = np.std(i1)
        i1 = (i1.copy() - mn2) / st2
        #i3 = (img - i1)*255
        #ret,i3 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        temp.append(img-i1)
        
    i2 = np.median(temp, axis = 0)
    print(i2[110:120,110:120])
#    i2 = cv2.GaussianBlur(i2.copy(),(3,3),0)
    #image thresholding since extra values need to be removed so that morph gradient image is smooth-no
    #ret,i2 = cv2.threshold(i2,0,1,cv2.THRESH_BINARY)
    #morphological transform
#    kernel = np.ones((3,3),np.uint8)
#    i2 = cv2.morphologyEx(i2.copy(), cv2.MORPH_GRADIENT, kernel)
    
    cv2.namedWindow(str(i))
    cv2.imshow(str(i), i2)
    #cv2.imwrite('C:\\Users\\Sanmoy\\Desktop\\pooja\\paper read\\sports\\dataset\\UIUC2\\'+str(i)+'.jpg',i2)
    k=cv2.waitKey()
    if k==97:
        continue
    else:
        break
cv2.destroyAllWindows()     
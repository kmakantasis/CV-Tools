# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import animation
import Farneback


def Img2Gray(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    img = np.array(img, dtype=np.uint8)
    
    return img
    
    

data = sio.loadmat('../../data/Classified/AIA_mac0001.SEQ_1.mat')
thermal_cube = data['thermal_cube']
del data

n_images = thermal_cube.shape[2]
img = np.transpose(thermal_cube[:,:,0])
newx,newy = img.shape[1]*2,img.shape[0]*2
img = cv2.resize(img,(newx,newy), interpolation=cv2.INTER_AREA)

prvs = Img2Gray(img)

farn = Farneback.Farneback()

fig = plt.figure()
ims = []

for i in range(n_images):
    frame2 = np.transpose(thermal_cube[:,:,i])
    frame2 = cv2.resize(frame2,(newx,newy), interpolation=cv2.INTER_AREA)
    nxt = Img2Gray(frame2)

    farn.CalculateOpticalFlow(prvs, nxt)
    rgb = farn.motion_image

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    
    prvs = nxt
    
    temp = np.array(farn.magnitude_image, dtype=np.float64)
    temp[temp >=32] = np.max(frame2)
    temp[temp < 32] = np.min(frame2)    
    
    temp2 = np.concatenate((frame2, temp), axis=1)        
    
    im = plt.imshow(temp2)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)

plt.show()

cv2.destroyAllWindows()
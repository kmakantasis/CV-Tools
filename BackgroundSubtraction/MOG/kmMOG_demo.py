# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import animation
import cv2
import MOG


data = sio.loadmat('../../data/Classified/AIA_mac0001.SEQ_1.mat')
thermal_cube = data['thermal_cube']
del data

n_images = thermal_cube.shape[2]

img = np.transpose(thermal_cube[:,:,0])

MOG_obj = MOG.MOGBS(history=10, nmixtures=15, backgroundRatio=0.9, noiseSigma=0.0, learningRate=0.001)

fig = plt.figure()
ims = []

for i in range(n_images): 
   img=np.transpose(thermal_cube[:,:,i])
   #img = img /  np.max(img)
   img = img - np.min(img)
   img = img / np.max(img)
   img = img * 255
   
   newx,newy = img.shape[1]*2,img.shape[0]*2
   img = cv2.resize(img,(newx,newy), interpolation=cv2.INTER_AREA)
   
   img_temp = np.array(img, dtype=np.uint8)
   MOG_obj.ApplyBS(img_temp)
   
   fgmask = MOG_obj.fg_image
       
   cv2.imshow('track',fgmask)
   cv2.waitKey(10)
   
   im = plt.imshow(fgmask)
   ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=10)

plt.show()
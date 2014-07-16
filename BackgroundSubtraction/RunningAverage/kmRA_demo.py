# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import animation
import RunningAverage


data = sio.loadmat('../../data/Classified/AIA_mac0001.SEQ_1.mat')
thermal_cube = data['thermal_cube']
del data

n_images = thermal_cube.shape[2]

img = np.transpose(thermal_cube[:,:,0])

# history of 10 seconds for 7.5 fps --> weight = 0.013
ra = RunningAverage.RunningAverageBS(weight=0.013) 

ra.ApplyBS(img)

bg = ra.bg_image
fg = ra.fg_image


fig = plt.figure()
ims = []

total = np.concatenate((bg, fg), axis=1)

for i in range(n_images):
    img = np.transpose(thermal_cube[:,:,i])
    ra.ApplyBS(img)

    bg = ra.bg_image
    fg = ra.fg_image
    
    fg[fg >=1.25] = np.max(img)
    fg[fg < 1.25] = np.min(img)    
    
    temp = np.concatenate((bg, img), axis=1)    
    temp = np.concatenate((temp, fg), axis=1)    
    
    im = plt.imshow(temp)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)

plt.show()
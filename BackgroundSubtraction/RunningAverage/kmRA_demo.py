# -*- coding: utf-8 -*-

import numpy as np
import cv2
import scipy.io as sio
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import animation
import RunningAverage
import Morphology


data = sio.loadmat('../../data/Classified/AIA_mac0001.SEQ_1.mat')
thermal_cube = data['thermal_cube']
del data

n_images = thermal_cube.shape[2]

img = np.transpose(thermal_cube[:,:,0])
#newx,newy = img.shape[1]*2, img.shape[0]*2
#img = cv2.resize(img,(newx,newy), interpolation=cv2.INTER_AREA)

# history of 10 seconds for 7.5 fps --> weight = 0.013
ra = RunningAverage.RunningAverageBS(weight=0.013) 

ra.ApplyBS(img)

bg = ra.bg_image
fg = ra.fg_image


fig = plt.figure()
ims = []

total = np.concatenate((bg, fg), axis=1)

bin_fg = np.zeros(shape=(fg.shape[0], fg.shape[1]), dtype=np.int32)
struct1 = ndimage.generate_binary_structure(2, 1)


for i in range(n_images):
    img = np.transpose(thermal_cube[:,:,i])
    #img = cv2.resize(img,(newx,newy), interpolation=cv2.INTER_AREA)
    ra.ApplyBS(img)

    bg = ra.bg_image
    fg = ra.fg_image
    
    bin_fg[fg < 1.25] = 0     
    bin_fg[fg >=1.25] = 1

    bin_fg = ndimage.binary_dilation(bin_fg, structure=struct1, iterations=3).astype(bin_fg.dtype)
    
    labels, nb_labels = Morphology.ConnenctedComponents(bin_fg)    
    filt_labels, areas, nb_new_labels = Morphology.FilterArea(bin_fg, labels, nb_labels, 150)
    rois = Morphology.DrawRectangle(np.asarray(filt_labels, dtype=np.int32), img, nb_new_labels, color=(305,0,0))
    
    
    fg[fg >=1.25] = np.max(img)
    fg[fg < 1.25] = np.min(img)    
    
    temp = np.concatenate((bg, img), axis=1)    
    temp = np.concatenate((temp, fg), axis=1)    
    
    im = plt.imshow(temp)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
ani.save('../../results/runnimg_average.mp4', fps=25, dpi=300)
plt.show()
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import animation
import SBGM


data = sio.loadmat('../../data/Classified/AIA_mac0001.SEQ_1.mat')
thermal_cube = data['thermal_cube']
del data

n_images = thermal_cube.shape[2]

n_hist = 100

sh = (thermal_cube.shape[1], thermal_cube.shape[0], n_hist)

sbgm = SBGM.SBGM(sh, method='mean')

fig = plt.figure()
ims = []

for i in range(n_images):
    img = np.transpose(thermal_cube[:,:,i])
    
    if i < n_hist:
        sbgm.ConstructHistory(img)
        
    else:
        sbgm.ConstructHistory(img)
        sbgm.SubtractBG()
        sbgm.ExtractDiffFG(img)
        
        bg_image = sbgm.bg_image
        fg_image = sbgm.fg_diff_image
        
        im = plt.imshow(fg_image)
        ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)

plt.show()


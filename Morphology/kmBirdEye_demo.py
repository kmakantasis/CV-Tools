# -*- coding: utf-8 -*-

import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt



data = sio.loadmat('../data/Classified/AIA_mac0001.SEQ_1.mat')
thermal_cube = data['thermal_cube']
del data

n_images = thermal_cube.shape[2]

img = np.transpose(thermal_cube[:,:,0])

pts1 = np.float32([[260,240],[15,240],[190,0],[140,0]])
pts2 = np.float32([[200,240],[100,240],[190,0],[140,0]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(320,240))

dst[dst<np.min(img)] = np.min(img)

fig = plt.figure()

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

#fig = plt.figure()
#im = plt.imshow(img, cmap = cm.Greys_r)
#plt.show()
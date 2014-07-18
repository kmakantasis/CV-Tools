# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import Morphology


img = cv2.imread('../data/binary.png', 0)

ret, img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)

labels, nb_labels = Morphology.ConnenctedComponents(img)

filt_labels, areas = Morphology.FilterArea(img, labels, nb_labels, 3100)

rois = Morphology.DrawRectangle(labels, nb_labels, color=(6,0,0))

temp = np.concatenate((labels, rois), axis=1)

fig = plt.figure()
im = plt.imshow(temp)
plt.show()

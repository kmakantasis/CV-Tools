# -*- coding: utf-8 -*-

import SURF
import numpy as np
import cv2

img1 = cv2.imread('../../data/alpha.png')
img2 = cv2.imread('../../data/alpha.png')

surf1 = SURF.SURF_Obj(img1)
surf1.SURF_Keypoints_Descriptors(plot_flag = False)

surf2 = SURF.SURF_Obj(img2)
surf2.SURF_Keypoints_Descriptors(plot_flag = False)

matches, distances = SURF.kmSURFMAtches(surf1.descriptors, surf2.descriptors, knn = 10)

fMatches = np.transpose(np.vstack((range(surf2.descriptors.shape[0]), matches[:,0])))

SURFCoord = SURF.kmPlotSURFMatches(np.hstack((img1,img2)), surf1.keyPoints, surf2.keyPoints, fMatches, img2.shape[1])
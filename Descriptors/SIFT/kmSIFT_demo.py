# -*- coding: utf-8 -*-

import SIFT
import numpy as np
import cv2

img1 = cv2.imread('../../data/CarData/TrainImages/pos-1.pgm')
img2 = cv2.imread('../../data/CarData/TrainImages/pos-1.pgm')

sift1 = SIFT.SIFT_Obj(img1)
sift1.SIFT_Keypoints_Descriptors(plot_flag = False)

sift2 = SIFT.SIFT_Obj(img2)
sift2.SIFT_Keypoints_Descriptors(plot_flag = False)

matches, distances = SIFT.kmSIFTMAtches(sift1.descriptors, sift2.descriptors, knn = 10)

fMatches = np.transpose(np.vstack((range(sift2.descriptors.shape[0]), matches[:,0])))

SIFTCoord = SIFT.kmPlotSIFTMatches(np.hstack((img1,img2)), sift1.keyPoints, sift2.keyPoints, fMatches, img2.shape[1])
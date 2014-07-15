# -*- coding: utf-8 -*-

import ImplicitShapeModel
import numpy as np
import cv2

ISM = ImplicitShapeModel.ISM("../../data/CarData/TrainImages/pos-", 100, 0.2, descriptor_type = 'surf',
                                                                             criterion = 'distance',
                                                                             method = 'complete',
                                                                             metric = 'correlation',
                                                                             use_raw = True)
ISM.ComputeDescriptors()
training_descriptors = ISM.training_descriptors
ISM.GenerateCodebook()
prototypes = ISM.prototypes
ISM.GenerateModel()
model = ISM.model
ISM.GenerateVote("../../data/CarData/TrainImages/pos-0.pgm", plot=False)
V = ISM.V            

#img = cv2.imread("../../data/CarData/TrainImages/pos-0.pgm", 0)
#det = np.zeros((img.shape[0],img.shape[1]))
#for i in range(len(V)):
#    if V[i][0] < 0 or V[i][0] > img.shape[0] or V[i][1] < 0 or V[i][1] > img.shape[1]:
#        continue
#    else:
#        det[int(V[i][0]), int(V[i][1])] = det[int(V[i][0]), int(V[i][1])] + V[i][2]
#
#det = (det / det.max()) * 255.0
#det = det.astype(int)
#
#
#import matplotlib.pyplot as plt
#
#plt.imshow(img, aspect='equal')
#plt.show()

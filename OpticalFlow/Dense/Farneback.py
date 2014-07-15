# -*- coding: utf-8 -*-

import numpy as np
import cv2


class Farneback(object):
    def __init__(self,  pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=1.2, poly_sigma=0):
        
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        
        self.motion_image = None
        self.magnitude_image = None
        
    
    def CalculateOpticalFlow(self, prev, nxt):
        
        hsv = np.zeros((prev.shape[0], prev.shape[1], 3))
        hsv[...,1] = 255
        
        flow = cv2.calcOpticalFlowFarneback(prev,nxt, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv = np.array(hsv, dtype=np.uint8)
        self.motion_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        self.magnitude_image = hsv[...,2]


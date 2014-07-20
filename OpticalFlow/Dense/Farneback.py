# -*- coding: utf-8 -*-

import numpy as np
import cv2


class Farneback(object):
    def __init__(self,  pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=1.2, poly_sigma=0):
        """               
            Function definition
            +++++++++++++++++++
            
            .. py:function:: __init__(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=1.2, poly_sigma=0)

                Initializes the object that computes a dense optical flow using the Gunnar Farneback’s algorithm.

                :param float pyr_scale: scaling factor between images in a Gaussian pyramid is subsequent levels.
                :param int levels: levels of the Gaussian pyramid.
                :param int winsize: averaging window size; larger values increase the algorithm robustness 
                                    to image noise and give more chances for fast motion detection, but 
                                    yield more blurred motion field.
                :param int iterations: number of iteration the algorithm does at each pyramid level.
                :param float poly_n: size of the pixel neighborhood used to find polynomial expansion in 
                                     each pixel; larger values mean that the image will be approximated 
                                     with smoother surfaces, yielding more robust algorithm and more 
                                     blurred motion field, typically poly_n =5 or 7.
                :param float poly_sigma: standard deviation of the Gaussian that is used to smooth 
                                         derivatives used as a basis for the polynomial expansion; for 
                                         poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value 
                                         would be poly_sigma=1.5.
                
            **Outputs - Class members**:
            
            * motion_image: an 3-d numpy array containing the motion image. The value of each pixel is \
                            assigned using the magnitude and the direction of the motion.
            * magnitude_image: an 2-d numpy array containing the magnitude of the motion at each one \
                               pixel normalized in :math:`[0,255]`.
            * direction_image: an 2-d numpy array containing the direction of the motion at each one \
                               pixel in degrees.
               
        """
        
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        
        self.motion_image = None
        self.magnitude_image = None
        self.direction_image = None
        
    
    def CalculateOpticalFlow(self, prev, nxt):
        """               
            Function definition
            +++++++++++++++++++
            
            .. py:function:: CalculateOpticalFlow(prev, nxt)
            
                Computes a dense optical flow using the Gunnar Farneback’s algorithm using two subsequent
                frames.

                :param numpy_array prev: the first frame of the two subsequent frames.
                :param numpy_array nxt: the second frame of the two subsequent frames.
                             
        """
        
        hsv = np.zeros((prev.shape[0], prev.shape[1], 3))
        hsv[...,1] = 255
        
        flow = cv2.calcOpticalFlowFarneback(prev,nxt, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.flow = flow

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=1)
        hsv[...,0] = ang/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv = np.array(hsv, dtype=np.uint8)
        self.motion_image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        self.magnitude_image = hsv[...,2]
        self.direction_image = ang


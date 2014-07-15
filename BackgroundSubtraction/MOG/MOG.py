# -*- coding: utf-8 -*-

import cv2


class MOGBS(object):
    def __init__(self, history=500, nmixtures=5, backgroundRatio=0.3, noiseSigma=0.0, learningRate=0.0):
        
        self.history = history
        self.nmixtures = nmixtures
        self.backgroundRatio = backgroundRatio
        self.noiseSigma = noiseSigma
        self.learningRate = learningRate
        
        self.fg_image = None
        
        self.bg = cv2.BackgroundSubtractorMOG(self.history, 
                                              self.nmixtures, 
                                              self.backgroundRatio, 
                                              self.noiseSigma)
                                              
                                              
    def ApplyBS(self, img):
            
        self.fg_image = self.bg.apply(img, None, self.learningRate)
        
        
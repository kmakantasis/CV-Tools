# -*- coding: utf-8 -*-

import cv2


class MOGBS(object):
    def __init__(self, history=500, nmixtures=5, backgroundRatio=0.3, noiseSigma=0.0, learningRate=0.0):
        """
            Function definition
            +++++++++++++++++++
            
            .. py:function:: __init__(history=500, nmixtures=5, backgroundRatio=0.3, noiseSigma=0.0, learningRate=0.0)

                Initializes the parameters of the background model.

                :param int history: length of frames history
                :param int nmixtures: number of Gaussian mixtures to be used during background modeling
                :param float backgroundRatio: Threshold defining whether the component is significant 
                                              enough to be included into the background model
                :param float noiseSigma: noise strength
                :param float learningRate: float parameter used for updating the background model
                
            **Outputs - Class members**:
            
            * fg_image: an 2-d numpy array containing the foreground_mask.
                
        """
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
        """            
            Function definition
            +++++++++++++++++++
            
            .. py:function:: ApplyBS(img)

                Updates the background model and calculates the foreground mask.

                :param numpy_array img: image on which the foreground mask is calculated
                
            The foreground mask can be accessed by **background_object_model.fg_image** parameter.
                
        """
            
        self.fg_image = self.bg.apply(img, None, self.learningRate)
        
        
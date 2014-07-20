# -*- coding: utf-8 -*-

import cv2

class RunningAverageBS(object):
    def __init__(self, weight=0.05):
        """
            Theory
            ++++++
            The running average backgroumd subtraction method creates a statistical background model
            at time instant :math:`t` using the following relation
            
            .. math::      
            
               bgModel(t) = a \cdot currentFrame + (1-a) \cdot bgModel(t-1)
               
            the parameter :math:`a` corresponds to the weight parameter of the algorithm.
               
            Function definition
            +++++++++++++++++++
            
            .. py:function:: __init__(weight=0.05)

                Initializes the parameters of the background model.

                :param float weight: controls the updating rate of background model
                
            **Outputs - Class members**:
            
            * bg_image: an 2-d numpy array containing the background model.
            * fg_image: an 2-d numpy array containing the foreground mask.
               
        """
        
        self.weight = weight
        
        self.bg_image = None
        self.fg_image = None


    def SubtractBG(self, img):
        """
            Function definition
            +++++++++++++++++++
            
            .. py:function:: SubtractBG(img)

                Initializes the class member **bg_image** which represents the background model.

                :param numpy_array img: image on which the background subtraction algorithm is applied.               
        """
        
        if self.bg_image == None:
            self.bg_image = img
            
        else:
            beta = 1 - self.weight
            alpha = self.weight
            gamma = 0.0
            self.bg_image = cv2.addWeighted(img, alpha, self.bg_image, beta, gamma)
            
            
    def ExtractFG(self, img):
        """
            Function definition
            +++++++++++++++++++
            
            .. py:function:: ExtractFG(img)

                Initializes the class member **fg_image** which represents the foreground mask.

                :param numpy_array img: image on which the background subtraction algorithm is applied.               
        """
        
        if self.bg_image == None:
            self.fg_image = img
            
        else:
            self.fg_image = cv2.absdiff(img, self.bg_image)
    
    
    def ApplyBS(self, img):
        """
            Function definition
            +++++++++++++++++++
            
            .. py:function:: ApplyBS(img)

                Applies SubtractBG(img) and ExtractFG(img) methods and initializes class members
                **bg_image** which represents the background model and **fg_image** which represents 
                the foreground mask 

                :param numpy_array img: image on which the background subtraction algorithm is applied.               
        """
        
        self.SubtractBG(img)
        self.ExtractFG(img)
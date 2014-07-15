# -*- coding: utf-8 -*-

import cv2

class RunningAverageBS(object):
    def __init__(self, weight=0.05):
        
        self.weight = weight
        
        self.bg_image = None
        self.fg_image = None


    def SubtractBG(self, img):
        
        if self.bg_image == None:
            self.bg_image = img
            
        else:
            cv2.accumulateWeighted(img, self.bg_image, self.weight)
            
            
    def ExtractFG(self, img):
        
        if self.bg_image == None:
            self.fg_image = img
            
        else:
            self.fg_image = cv2.absdiff(img, self.bg_image)
    
    
    def ApplyBS(self, img):
        
        self.SubtractBG(img)
        self.ExtractFG(img)
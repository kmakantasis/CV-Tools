# -*- coding: utf-8 -*-

import numpy as np
import cv2


class SBGM(object):
    def __init__(self, hist_shape, method='median', sigma_hat=5, stat_thresh=5, diff_thresh=1.25):
        
        self.hist_shape = hist_shape
        self.n_hist = hist_shape[2]
        self.construction_counter = 0
        self.update_counter = 0
        self.sigma_hat = sigma_hat
        self.stat_thresh = stat_thresh ** 2
        self.diff_thresh = diff_thresh
        
        if method == 'median':
            self.method = np.median
        elif method == 'mean':
            self.method = np.mean

        self.history = np.zeros(hist_shape)
        self.bg_image = None
        self.fg_stat_image = None
        self.fg_diff_image = None

        
    def ConstructHistory(self, img):
        
        if self.construction_counter < self.n_hist:
            self.history[...,self.construction_counter] = img
            self.construction_counter = self.construction_counter + 1
            
        else:
            self.history[...,self.update_counter] = img
            self.update_counter = self.update_counter + 1
            
            if self.update_counter == self.n_hist:
                self.update_counter = 0
            
    
    def SubtractBG(self, img):
               
        if self.construction_counter == self.n_hist:
            self.bg_image = self.method(self.history, axis=2)
            
            
    def ExtractStatFG(self, img):
        
        if self.construction_counter == self.n_hist:
            w = np.zeros(self.hist_shape)
            i_diff_sq = np.zeros(self.hist_shape)
            
            for i in range(self.n_hist):
                i_diff = self.history[...,i] - self.bg_image
                nom = i_diff ** 2
                denom = -2 * (self.sigma_hat ** 2)
                w[...,i] = np.exp(nom / denom)
                
            mu_nom = np.sum(w * self.history, axis=2)
            denom = np.sum(w, axis=2)
            mu = mu_nom / denom

            for i in range(self.n_hist):
                i_diff = self.history[...,i] - mu 
                i_diff_sq[...,i] = i_diff ** 2
                
            
            sigma_nom = np.sum(w * i_diff_sq, axis=2)   
            sigma = sigma_nom / denom
            
            D = ((img - mu) ** 2) / sigma
            
            D[D < self.stat_thresh] = 0
            D[D >= self.stat_thresh] = 1
            
            self.fg_stat_image = D
            
            
    def ExtractDiffFG(self, img):
        
        fg = cv2.absdiff(img, self.bg_image)
        fg[fg < self.diff_thresh] = 0
        fg[fg >= self.diff_thresh] = 1
        
        self.fg_diff_image = fg
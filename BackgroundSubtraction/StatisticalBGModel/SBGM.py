# -*- coding: utf-8 -*-

import numpy as np
import cv2


class SBGM(object):
    def __init__(self, hist_shape, method='median', sigma_hat=5, stat_thresh=5, diff_thresh=1.25):
        """
            Theory
            ++++++
            This fucntion initializes a stastistical background modeling object. To be more specific, it uses
            a history of :math:`N` frames and computes a representative image for the background. This image
            can be the **median** or the **mean** image of the :math:`N` frames. Then, the statistical background
            model for each pixel is created by computing weighted means and variances from the :math:`N` frames
            using the following relations
            
            .. math::      
            
               \mu(x,y) = \\frac{\sum_{i=1}^N w_i(x,y) \cdot I_i(x,y)}{\sum_{i=1}^N x_i(x,y)}
               
            for the mean values and
            
            .. math::      
            
               \sigma^2(x,y) = \\frac{\sum_{i=1}^N w_i(x,y) \cdot (I_i(x,y) - \mu(x,y))^2}{\\frac{N-1}{N}\sum_{i=1}^N x_i(x,y)}
               
            for the varriance. The weights :math:`w_i(x,y)` are used to minimize the effect of outliers and 
            computed from a Gaussian distribution centered at the representative image :math:`I_r` using the
            following relation
            
            .. math::      
            
               w_(x,y) = exp\\frac{(I_i(x,y)-I_r(x,y))^2}{-2 \hat{\sigma}^2}
               
            the parameter :math:`\hat{\sigma}` is set by the user. Once the background model has been estimated,
            foreground pixels for a new input image :math:`I` are obtained using the square Mahalanobis 
            distance
            
            .. math::      
            
               D(x,y) = foreground \:\:\: if \:\: \\frac{(I(x,y)-\mu(x,y))^2}{\sigma(x,y)^2} > T
               
            parameter :math:`T` is another user defined parameter, it can be set to the same value with the
            parameter :math:`\hat{\sigma}`.
               
            Function definition
            +++++++++++++++++++
            
            .. py:function:: __init__(hist_shape, method='median', sigma_hat=5, stat_thresh=5, diff_thresh=1.25)

                Initializes the parameters of the background model.

                :param tuple hist_shape: contains the image height and width and the number of frames in history.
                :param string method: it can be set to 'median' or 'mean' to create the representative image.
                :param float sigma_hat: corresponds to :math:`\hat{\sigma}` parameter.
                :param float stat_thresh: corresponds to :math:`T` parameter.
                :param float diff_thresh: it is used with the method *ExtractDiffFG()*.
                
            **Outputs - Class members**:
            
            * history: an 3-d numpy array the :math:`N` frames of the history.
            * bg_image: an 2-d numpy array containing the background model.
            * fg_stat_image: an 2-d numpy array containing the foreground mask.
            * fg_diff_image: an 2-d numpy array containing the foreground mask.It is created by *ExtractDiffFG()* method.
               
        """
        
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
        """         
            Function definition
            +++++++++++++++++++
            
            .. py:function:: ConstructHistory(img)

                Constructs and updates the numpy array that holds the frames history

                :param numpy _array img: image that is used during history construction or updating.
                
            **Outputs - Class members**:
            
            * history: an 3-d numpy array the :math:`N` frames of the history.               
        """
        
        if self.construction_counter < self.n_hist:
            self.history[...,self.construction_counter] = img
            self.construction_counter = self.construction_counter + 1
            
        else:
            self.history[...,self.update_counter] = img
            self.update_counter = self.update_counter + 1
            
            if self.update_counter == self.n_hist:
                self.update_counter = 0
            
    
    def SubtractBG(self):
        """         
            Function definition
            +++++++++++++++++++
            
            .. py:function:: SubtractBG()

                Calculates the statistical background model.

                :param None None: It takes no inputs.
                
            **Outputs - Class members**:
            
            * bg_image: an 2-d numpy array containing the background model.            
        """
               
        if self.construction_counter == self.n_hist:
            self.bg_image = self.method(self.history, axis=2)
            
            
    def ExtractStatFG(self, img):
        """         
            Function definition
            +++++++++++++++++++
            
            .. py:function:: ExtractStatFG(img)

                Calculates the foreground mask on a new captured image.

                :param numpy _array img: image on which the foreground mask is calculated.
                
            **Outputs - Class members**:
            
            * fg_stat_image: an 2-d numpy array containing the foreground mask.            
        """
        
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
        """         
            Function definition
            +++++++++++++++++++
            
            .. py:function:: ExtractDiffFG(img)

                Calculates the foreground mask on a new captured image.

                :param numpy _array img: image on which the foreground mask is calculated.
                
            **Outputs - Class members**:
            
            * fg_diff_image: an 2-d numpy array containing the foreground mask.            
        """
        
        fg = cv2.absdiff(img, self.bg_image)
        fg[fg < self.diff_thresh] = 0
        fg[fg >= self.diff_thresh] = 1
        
        self.fg_diff_image = fg
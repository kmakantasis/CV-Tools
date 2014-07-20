# -*- coding: utf-8 -*-

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import label


def ConnenctedComponents(img):
    """
        Function definition
        +++++++++++++++++++
            
        .. py:function:: ConnenctedComponents(img)
            
            This method is used for finding and labeling connected components in a binary image. 

            :param numpy_array img: binary image on which connected components need to be found.
            :return: labeled image and number of labels.
            :rtype: numpy array of the same dimensions with img - int.
        """
        
    all_labels = label(img)
    
    return all_labels
    
    
def FilterArea(img, labels, nb_labels, threshold):
    
    """
        Function definition
        +++++++++++++++++++
            
        .. py:function:: FilterArea(img, labels, nb_labels, threshold)
            
            This method eliminates connected components whose areas are smaller than a threshold. 

            :param numpy_array img: binary image on which connected components were estimated.
            :param numpy_array img: labeled image that corresponds to the output of the **ConnenctedComponents(img)**
                                    method.
            :return: labeled omage.
            :rtype: numpy array of the same dimensions with img.
        """
    
    filt_labels = np.copy(labels)
    
    areas = ndimage.sum(img, labels, range(nb_labels + 1))
    mask_size = areas < threshold
    remove_pixel = mask_size[labels]
    filt_labels[remove_pixel] = 0
    new_labels = np.unique(filt_labels)
    filt_labels = np.searchsorted(new_labels, filt_labels)
        
    return filt_labels, new_labels.shape[0]-1
    
    
def DrawRectangle(img, dst, nb_labels, color=(255,0,0)):
    
    rois =   np.copy(img)  
    
    for i in range(nb_labels):
        slice_x, slice_y = ndimage.find_objects(rois==i+1)[0]
        cv2.rectangle(dst,(slice_y.start,slice_x.start),(slice_y.stop,slice_x.stop),color,1)
        
    return rois

        
    
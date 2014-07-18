# -*- coding: utf-8 -*-

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import label


def ConnenctedComponents(img):
    
    # It uses a binary image with zeros for background and ones for foreground
    
    all_labels = label(img)
    
    return all_labels
    
    
def FilterArea(img, labels, nb_labels, threshold):
    
    # it uses a labeled image derived from ConnectedComponents method
    
    filt_labels = np.copy(labels)
    
    areas = ndimage.sum(img, labels, range(nb_labels + 1))
    mask_size = areas < threshold
    remove_pixel = mask_size[labels]
    filt_labels[remove_pixel] = 0
    new_labels = np.unique(filt_labels)
    filt_labels = np.searchsorted(new_labels, filt_labels)
        
    return filt_labels, areas, new_labels.shape[0]-1
    
    
def DrawRectangle(img, nb_labels, color=(255,0,0)):
    
    rois =   np.copy(img)  
    
    for i in range(nb_labels):
        slice_x, slice_y = ndimage.find_objects(rois==i+1)[0]
        cv2.rectangle(rois,(slice_y.start,slice_x.start),(slice_y.stop,slice_x.stop),color,3)
        
    return rois

        
    
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pylab import *

class SURF_Obj(object):
    def __init__(self,img, upright = False):     
        """
        **Definition**: SURF_Obj(img, upright = False)
        
        SURF object class. Implementation of SURF detector and SURF descriptor.
        
        **Inputs**:
            * img: the source image for which the descriptors are computed
            * upright (optional): default *False*. If *True* the upright version \
            of SURF is used.
            
        **Outputs - Class members**:
            * keyPoints: extracted SURF keypoints for the image *img*. Type: KeyPoint \
            class of OpenCV.
            * descriptors: extracted SURF descriptors for image *img*. Type: numpy \
            array of dimensions *number_of_keypoints x 64*. Each row represents \
            a keypoint descriptor.
            * coordinates: the coordinates of the extracted keypoints.
        """
        self.input = img
        self.surfDetector = cv2.FeatureDetector_create("SURF")
        self.surfDetector.setInt("upright", upright)
        
        self.surfExtractor =  cv2.DescriptorExtractor_create("SURF")
        
        self.keyPoints = None
        self.descriptors = None
        self.coordinates = None
        
        
    def SURF_Keypoints_Descriptors(self, plot_flag = True):
        """
        **Definition**: SURF_Keypoints_Descriptors(plot_flag = True)
        
        Implements keypoint detection and descriptors computation. Detected keypoints \
        are stored in **self.keyPoints** and computed descriptors are stored in \
        **self.descriptors**. Additionally, it extracts keypoints coordinates. Coordinates \
        are stored in **self.coordinates**.
        
        **Inputs**:
            * plot_flag (optional): default *True*. If *True* this method plots keypoints \
            on source image *img*.
        
        **Outputs**: 
            * *None*
        """
        keyPoints = self.surfDetector.detect(self.input, None)
        keyPoints, descriptors = self.surfExtractor.compute(self.input, keyPoints)
       
        self.keyPoints = keyPoints
        self.descriptors = descriptors
        
        m = size(self.keyPoints)
        surfCoord = np.zeros([m,2])
        for i in range(m):
             surfCoord[i][0] = self.keyPoints[i].pt[1]
             surfCoord[i][1] = self.keyPoints[i].pt[0]
        
        if plot_flag == True:
            figure()
            gray()
            imshow(self.input)
            plot([p[1] for p in surfCoord], [p[0] for p in surfCoord], '*')
            axis('off')
            show()
 
        self.coordinates = surfCoord
        
 
def kmSURFMAtches(descriptors_dst, descriptors_src, knn = 5):
    """
    **Definition**: kmSURFMAtches(descriptors_dst, descriptors_src, knn = 5)
    
    Computes the matches between two different sets of SURF descriptors. Specifically, \
    it computes the matches from *descriptors_src* to *descriptors_dst* and find the *knn* \
    best matches.
    
    **Inputs**:
            * descriptors_dst: set of descriptors **to** match.
            * descriptors_src: set of descriptors **from** which the matching process is performed.
            * knn (optional): default *5*. *knn* denotes how many matches betweent the \
            two sets of descriptors will be computed.
        
    **Outputs**: 
            * idx: an *number_of_descriptors_src x knn* matrix. The i-th row contains \
            the *knn* nearest neigbors for descriptor i of descriptors_src set.
            * dist: an *number_of_descriptors_src x knn* matrix. The i-th row contains \
            the distances to the *knn* nearest neigbors for descriptor i of descriptors_src set.
    """
    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(descriptors_dst, flann_params)
    idx, dist = flann.knnSearch(descriptors_src, knn, params={})
    
    del flann    
    
    return idx, dist       
     
 
def kmPlotSURFMatches(img, kp1, kp2, fMatches, disp):
    """
    **Definition**: kmPlotSURFMatches(img, kp1,, kp2, fMatches, disp)
    
    Plots correspondent points between two different sets of SURF descriptors.
    
    **Inputs**:
            * img: the image on which the plotting is performed.
            * kp1: detected keypoins of the first image.
            * kp2: detected keypoins of the second image.
            * fMatches: an *number_of_correspondences x 2* matrix containing the correspondent pairs.
            * disp: displacement betweent img1 and img2 on img. Usually it is equal to \
            img2.shape[1]
        
    **Outputs**: 
            * SURFCoord:  an *number_of_correspondences x 4* matrix containing the coordinates \
            of corespondent keypoints.
            
    **Usage example**:

        fMatches = np.transpose(np.vstack((range(surf2.descriptors.shape[0]), matches[:,0]))) 
    
        img = np.hstack((img1,img2)) 
    
        SURFCoord = kmPlotSURFMatches(img, kp1, kp2, fMatches, img2.shape[1])        
    """
    SURFCoord = []   
    for p in fMatches:
        img1Coord = kp1[int(p[0])].pt
        img2Coord = kp2[int(p[0])].pt
        img12Coord = np.hstack((img1Coord, img2Coord))
        SURFCoord = np.append(SURFCoord, img12Coord)
        
    SURFCoord = np.reshape(SURFCoord, (-1,4))
    
    figure()
    gray()
    imshow(img)
    
    for p in SURFCoord:
        plot((p[0], p[2]+disp), (p[1], p[3]), 'g')

    axis('off')
    show()
    return SURFCoord
    

    
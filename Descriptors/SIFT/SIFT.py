# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pylab import *

class SIFT_Obj(object):
    def __init__(self,img, nOctaveLayers = 3, 
                           contrastThreshold = 0.04, 
                           edgeThreshold = 10, 
                           sigma = 1.6):     
        """
        **Definition**: SIFT_Obj(img, upright = False)
        
        SIFT object class. Implementation of SIFT detector and SIFT descriptor.
        
        **Inputs**:
            * img: the source image for which the descriptors are computed
            * nOctaveLayers (optional): default *3*. See OpenCV documentation
            * contrastThreshold (optional): default *0.04*. See OpenCV documentation
            * edgeThreshold (optional): default *10*. See OpenCV documentation
            * sigma (optional): default *1.6*. See OpenCV documentation
            
        **Outputs - Class members**:
            * keyPoints: extracted SIFT keypoints for the image *img*. Type: KeyPoint \
            class of OpenCV.
            * descriptors: extracted SIFT descriptors for image *img*. Type: numpy \
            array of dimensions *number_of_keypoints x 128*. Each row represents \
            a keypoint descriptor.
            * coordinates: the coordinates of the extracted keypoints.
        """
        self.input = img
        self.siftDetector = cv2.FeatureDetector_create("SIFT")
        self.siftDetector.setInt("nOctaveLayers", nOctaveLayers)
        self.siftDetector.setDouble("contrastThreshold", contrastThreshold)
        self.siftDetector.setInt("edgeThreshold", edgeThreshold)
        self.siftDetector.setDouble("sigma", sigma)
        
        self.siftExtractor =  cv2.DescriptorExtractor_create("SIFT")
        
        self.keyPoints = None
        self.descriptors = None
        self.coordinates = None
        
        
    def SIFT_Keypoints_Descriptors(self, plot_flag = True):
        """
        **Definition**: SIFT_Keypoints_Descriptors(plot_flag = True)
        
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
        keyPoints = self.siftDetector.detect(self.input, None)
        keyPoints, descriptors = self.siftExtractor.compute(self.input, keyPoints)
       
        self.keyPoints = keyPoints
        self.descriptors = descriptors
        
        m = size(self.keyPoints)
        siftCoord = np.zeros([m,2])
        for i in range(m):
             siftCoord[i][0] = self.keyPoints[i].pt[1]
             siftCoord[i][1] = self.keyPoints[i].pt[0]
        
        if plot_flag == True:
            figure()
            gray()
            imshow(self.input)
            plot([p[1] for p in siftCoord], [p[0] for p in siftCoord], '*')
            axis('off')
            show()
 
        self.coordinates = siftCoord
        
 
def kmSIFTMAtches(descriptors_dst, descriptors_src, knn = 5):
    """
    **Definition**: kmSIFTMAtches(descriptors_dst, descriptors_src, knn = 5)
    
    Computes the matches between two different sets of SIFT descriptors. Specifically, \
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
     
 
def kmPlotSIFTMatches(img, kp1, kp2, fMatches, disp):
    """
    **Definition**: kmPlotSIFTMatches(img, kp1,, kp2, fMatches, disp)
    
    Plots correspondent points between two different sets of SIFT descriptors.
    
    **Inputs**:
            * img: the image on which the plotting is performed.
            * kp1: detected keypoins of the first image.
            * kp2: detected keypoins of the second image.
            * fMatches: an *number_of_correspondences x 2* matrix containing the correspondent pairs.
            * disp: displacement betweent img1 and img2 on img. Usually it is equal to \
            img2.shape[1]
        
    **Outputs**: 
            * SIFTCoord:  an *number_of_correspondences x 4* matrix containing the coordinates \
            of corespondent keypoints.
            
    **Usage example**:

        fMatches = np.transpose(np.vstack((range(sift2.descriptors.shape[0]), matches[:,0]))) 
    
        img = np.hstack((img1,img2)) 
    
        SIFTCoord = kmPlotSIFTMatches(img, kp1, kp2, fMatches, img2.shape[1])        
    """
    SIFTCoord = []   
    for p in fMatches:
        img1Coord = kp1[int(p[0])].pt
        img2Coord = kp2[int(p[0])].pt
        img12Coord = np.hstack((img1Coord, img2Coord))
        SIFTCoord = np.append(SIFTCoord, img12Coord)
        
    SIFTCoord = np.reshape(SIFTCoord, (-1,4))
    
    figure()
    gray()
    imshow(img)
    
    for p in SIFTCoord:
        plot((p[0], p[2]+disp), (p[1], p[3]), 'g')

    axis('off')
    show()
    return SIFTCoord
    

    
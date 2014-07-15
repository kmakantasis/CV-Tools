# -*- coding: utf-8 -*-

import cv2
import numpy as np
import cmath

def ComputeSCD(img_filename, ratio):
    """
        Computes and returns the shape context descriptors for an image. \n
        **Inputs**:
        
            * img_filename : the filename of the image \n
            * ratio : defines the edge points sampling procedure \n
        
        
        **Outputs**:
        
            * descriptors_list : list of dictionaries each one contains the decriptor \
            for an edge point along with the coordinates of that point \n
            
        *External Links*
        
        1. http://en.wikipedia.org/wiki/Shape_context \n
        2. http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf \n
    """
            
    img = cv2.imread(img_filename)
    img_canny = cv2.Canny(img, 100, 200)
    edges = np.asarray(np.where(img_canny==255))

    high_val = edges.shape[1]-1
    samples = high_val/ratio
    edge_samples = np.random.random_integers(low=0, high=high_val, size=samples)
    edge_points = edges[:, edge_samples]

    vectors = [edge_points[:,i] - edge_points[:,j] for i in range(samples) for j in range(samples) if i != j]

    polar_vectors = np.asarray([cmath.polar(complex(vec[0], vec[1])) for vec in vectors])
    polar_vectors[:,1] = np.degrees(polar_vectors[:,1])

    p_mean = np.mean(polar_vectors[:,0])
    norm_polar_vectors = [(np.log(polar_vectors[i,0]/p_mean), polar_vectors[i,1]) for i in range(polar_vectors.shape[0])]
    
    x_min = np.log(0.125)
    x_max = np.log(2.0)
    x_range = x_max - x_min
    x_step = x_range/5
    x_epsilon = 0.0000001
    xedges = list(np.arange(x_min,x_max+x_epsilon, x_step))
    yedges = list(np.arange(-180,180+30, 30))

    split_list_idx = samples -1
    descriptors_list = []
    d_append = descriptors_list.append
    hist2d = np.histogram2d
    for i in range(samples):
        temp_list = SplitList(norm_polar_vectors, split_list_idx, i)
        H, xedges, yedges = hist2d(temp_list[:,0], temp_list[:,1], bins=(xedges, yedges), normed=False)
        H = H / (samples - 1.0)
        d1 = {'coordinates':edge_points[:,i], 'descriptor':H }
        d_append(d1)
    
    return descriptors_list
        
        
def SplitList(original_list, split_list_idx, index):
    start_idx = index * split_list_idx
    end_idx = start_idx + split_list_idx
    temp_list = np.asarray(original_list[start_idx:end_idx])
    
    return temp_list   
    
    
def Chi_Squared(d1, d2):
    nominator = (d1 - d2) ** 2
    denominator = d1 + d2
    temp = nominator / denominator
    temp[np.isnan(temp)] = 0
    
    return np.sum(temp) / 2.0
    
    
def CostMatrix(desc1, desc2):
    """
        Theory
        ++++++
        Shape context descriptors correspond to 2D histograms. The distance between two normalized
        histograms can be evaluated using the :math:`\chi^2 - test` from statistics. For two 
        different histograms :math:`h_i` and :math:`h_j` with :math:`k` bins it is defined as. 
        
        .. math::
        
            \chi^2 - test = \\frac{1}{2} \\sum_{k=1}^{K} \\frac{[h_i(k) - h_j(k)]^2}{h_i(k) + h_j(k)}
        
            
        Function definition
        +++++++++++++++++++
            
        .. py:function:: CostMatrix(desc1, desc2)

            Computes and returns the cost matrix containing the distances between every pair of the
            descriptors of two different shapes. The distance between two descriptors corresponds to 
            the :math:`\chi^2 - test`.

            :param list desc1: list of 2D arrays. Each array in the list corresponds to a descriptor
                               for a point in the first shape
            :param list desc2: list of 2D arrays. Each array in the list corresponds to a descriptor
                               for a point in the second shape
            :return: returns the cost matrix containing the distances between every pair of the
                     descriptors of two different shapes
            :rtype: 2D float array           
           
    """
    
    dl1 = len(desc1)
    dl2 = len(desc2)
    cost = np.zeros((dl1, dl2))        

    for i in range(dl1):
        d1 = desc1[i]
        for j in range(dl2):
            d2 = desc2[j]
            cost[i,j] = Chi_Squared(d1,d2)
            
    return cost 
    
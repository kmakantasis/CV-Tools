# -*- coding: utf-8 -*-

import scipy.cluster.hierarchy as sh
import numpy as np


class Agglomerative(object):
    def __init__(self, input, threshold):
        """
        **Definition**: Agglomerative(input, threshold)
        
        Agglomerative clustering object class. Implementation of Agglomerative clustering algorithm.
        
        **Inputs**:
            * input: a distance matrix in the form of that **pdist** returns or the raw \
            observations. In the latter case it is an *m x n* matrix, where *m* is the \
            number of observations and *n* are their features.
            * threshold: distance threshold, which is used during observations' agglomeration.
            
            
        **Outputs - Class members**:
            * clusters: an 1-d numpy array containing the cluster index for each observation.
        """
        self.input = input
        self.threshold = threshold
        
        self.clusters = None
        
        
    def AggloClustering(self, criterion = 'distance', method = 'single', metric = 'euclidean', normalized = False):
        """
        **Definition**: AggloClustering(criterion = 'distance', method = 'single')
        
        Agglomerative clustering algorithm implementation.
        
        **Inputs**:
            * criterion (optional): default 'distance'. This is used in combination with \
            threshold. More information can be found on scipy.cluster.hierarchy documentation.
            * method (optional): default 'single'. This argument defines the linkage algorithm. \
            More information can be found on scipy.cluster.hierarchy documentation.            
            
        **Outputs**:
            * clusters: an 1-d numpy array containing the cluster index for each observation.
        """
        
        if normalized == True:
            X = self.NormalizedData()
        else:
            X = self.input
            
        
        Z = sh.linkage(X, method = method, metric = metric)
        Z[np.where(Z <0)] = 0
        cl = sh.fcluster(Z, self.threshold, criterion = criterion)
        
        self.clusters = cl
        
            
    def NormalizedData(self):
        """
        **Definition**: NormalizedData()
        
        Data normalization. From each one datum subtracts its mean value.
        
        **Inputs**:
            * None (all necessary inputs are derived from Codebook object construction)             
            
        **Outputs**:
            * normX: normalized raw observations data.
        """
        X = self.input
        samples = X.shape[0]
        normX = np.asarray([X[i] - np.mean(X[i]) for i in range(samples)])
       
        return normX   
       
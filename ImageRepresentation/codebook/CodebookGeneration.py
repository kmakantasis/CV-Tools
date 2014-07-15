# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist
import AgglomerativeClustering


class Codebook(object):
    def __init__(self, X, clustering_method = 'agglomerative', **options):
        """
        **Definition**: Codebook(X, Y, clustering_method = 'agglomerative', **options)
        
        Codebook object class. Implementation of codebook generation algorithm.
        
        **Inputs**:
            * X: raw observation data.
            * clustering_method (optional): default *agglomerative*. Defines the \
            clustering method to be used for finding codebook's prototypes. Now, only \
            "agglomerative" option is supported.
            * options: input parameters for clustering algorithm.
            
                * threshold: criterion threshold to be used by agglomerative clustering \
                algorithm
                * criterion: This is used in combination with threshold. More information \
                can be found on scipy.cluster.hierarchy documentation.
                * method: This argument defines the linkage algorithm. More information \
                can be found on scipy.cluster.hierarchy documentation.
                * metric: metric to be used during distance calculation.
                *use_raw: boolean. Defines if claustering will use the raw obervations or \
                the distance matrix.
            
            
        **Outputs - Class members**:
            * prototypes: clusters' centroids. They are computed by using the mean value of \
            each cluster elements.
            * clusters: an 1-d numpy array containing the cluster index for each observation.
        """
        self.X = X
        self.Y = pdist(X, metric = options['metric'])

        self.clustering_method = clustering_method
        self.options = options
        
        self.prototypes = None
        self.clusters = None
            
    
    def AgglomerativeClustering(self):
        """
        **Definition**: AgglomerativeClustering()
        
        Method that uses agglomerative clustering algorithm to group together similar \
        observations.
        
        **Inputs**:
            * None (all necessary inputs are derived from Codebook object construction)
            
        **Outputs**:
            * clusters: an 1-d numpy array containing the cluster index for each observation.
        """
        options = self.options
        
        if options['use_raw'] == False:
            agglo = AgglomerativeClustering.Agglomerative(self.Y, options['threshold'])
        elif options['use_raw'] == True:
            agglo = AgglomerativeClustering.Agglomerative(self.X, options['threshold'])
        
        agglo.AggloClustering(criterion = options['criterion'], 
                              method = options['method'], 
                              metric = options['metric'])        
        clusters = agglo.clusters

        return clusters

    
    def PrototypesExtraction(self):
        """
        **Definition**: PrototypesExtraction()
        
        This method is based on the output of AgglomerativeClustering method to compute \
        codebook prototypes. Codebbok prototypes are represented by the mean value of \
        each cluster elements.
        
        **Inputs**:
            * None (all necessary inputs are derived from Codebook object construction)
            
        **Outputs**:
            * None (prototypes and clusters can be accessed implicitly through Codebook object)
            
                * prototypes: object.prototypes
                * clusters: object.clusters
                
        """
        if self.clustering_method == 'agglomerative':
            clusters = self.AgglomerativeClustering()  
            num_of_clusters = np.max(clusters)
            
            centroids = [np.mean(self.X[np.where(clusters==i+1)], axis = 0) for i in range(num_of_clusters)]            
            
            self.clusters = clusters
            self.prototypes = np.asarray(centroids)
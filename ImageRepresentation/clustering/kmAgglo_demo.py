# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import pdist
import AgglomerativeClustering

x1 = np.array(([1,2], [2,1], [1,0]))
x2 = np.array(([-1,1], [-1,0], [-2,-1]))

X = np.concatenate((x1, x2))

Y = pdist(X, 'correlation')

agglo = AgglomerativeClustering.Agglomerative(Y, 0.3)

# in order to use the centroid coordinates during clustering the method 
# argument must be set to 'centroid' and raw observation must be used instead
# of the matrix Y
agglo.AggloClustering(criterion = 'distance', method = 'single', metric = 'correlation', normalized = False)

clusters = agglo.clusters

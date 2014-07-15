# -*- coding: utf-8 -*-

import numpy as np
import CodebookGeneration

x1 = np.array(([1,2], [2,1], [1,0]))
x2 = np.array(([-1,1], [-1,0], [-2,-1]))

X = np.concatenate((x1, x2))

code = CodebookGeneration.Codebook(X, threshold = 0.3, 
                                      criterion = 'distance', 
                                      method = 'single', 
                                      metric = 'correlation', 
                                      use_raw = False)
code.PrototypesExtraction()

clusters = code.clusters
prototypes = code.prototypes

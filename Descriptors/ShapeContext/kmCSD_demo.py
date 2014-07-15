# -*- coding: utf-8 -*-

import time
import SCDescriptors

filename = '../../data/alpha.png'

t1 = time.time()
cs_descriptors = SCDescriptors.ComputeSCD(filename, 100)

desc1 = [cs_descriptors[i]['descriptor'] for i in range(len(cs_descriptors))]
desc2 = [cs_descriptors[i]['descriptor'] for i in range(len(cs_descriptors))]

cost = SCDescriptors.CostMatrix(desc1, desc2)

print time.time() - t1


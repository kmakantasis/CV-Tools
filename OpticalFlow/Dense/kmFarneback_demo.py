# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.io as sio
from scipy import ndimage
from scipy.spatial.distance import pdist
#from scipy.cluster.vq import kmeans,vq
import matplotlib.pyplot as plt
from matplotlib import animation
import Farneback
import Morphology
import AgglomerativeClustering


def Img2Gray(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    img = np.array(img, dtype=np.uint16)
    
    return img
    
  
def ClusterObjects(farn, struct_elem):
    magn_img = farn.magnitude_image
    dir_img = farn.direction_image    
    
    bin_img = np.zeros(shape=(magn_img.shape[0], magn_img.shape[1]), dtype=np.uint8)
    bin_img[magn_img < 25] = 0
    bin_img[magn_img >= 25] = 1
    
    bin_img = ndimage.binary_dilation(bin_img, structure=struct_elem, iterations=3).astype(bin_img.dtype)

    labels, nb_labels = Morphology.ConnenctedComponents(bin_img)    
    filt_labels, areas, nb_new_labels = Morphology.FilterArea(bin_img, labels, nb_labels, 480)
    
    
    temp_magn = ndimage.mean(magn_img, filt_labels, range(nb_new_labels + 1))
    temp_dir = ndimage.mean(dir_img, filt_labels, range(nb_new_labels + 1))
    
    data = np.concatenate((np.reshape(temp_magn, (-1,1)), np.reshape(temp_dir, (-1,1))), axis=1)
    
    clusters = -1
    if nb_new_labels >= 1:
        Y = pdist(data, 'euclidean')
        agglo = AgglomerativeClustering.Agglomerative(Y, 50.)
        agglo.AggloClustering(criterion = 'distance', method = 'single', metric = 'euclidean', normalized = False)

        clusters = agglo.clusters
             
    bin_img[filt_labels == 0] = 0
    bin_img[filt_labels >= 1] = 1
    
    
    
    return bin_img, nb_new_labels, temp_magn, temp_dir, data, clusters
    
    
  

data = sio.loadmat('../../data/Classified/AIA_mac0001.SEQ_1.mat')
thermal_cube = data['thermal_cube']
del data

n_images = thermal_cube.shape[2]
img = np.transpose(thermal_cube[:,:,0])
newx,newy = img.shape[1]*2,img.shape[0]*2
img = cv2.resize(img,(newx,newy), interpolation=cv2.INTER_AREA)

prvs = Img2Gray(img)

farn = Farneback.Farneback()

fig = plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ims = []

struct_elem = ndimage.generate_binary_structure(2, 1)

for i in range(n_images):
    frame2 = np.transpose(thermal_cube[:,:,i])
    frame2 = cv2.resize(frame2,(newx,newy), interpolation=cv2.INTER_AREA)
    nxt = Img2Gray(frame2)

    farn.CalculateOpticalFlow(prvs, nxt)
    rgb = farn.motion_image
    
    ######### Color Quantization #####################################
    #temp_rgb = np.reshape(rgb,(rgb.shape[0]*rgb.shape[1],3))
    #centroids,_ = kmeans(temp_rgb,6) 
    #qnt,_ = vq(temp_rgb,centroids)  
    #centers_idx = np.reshape(qnt,(rgb.shape[0],rgb.shape[1]))
    #clustered = centroids[centers_idx]
    ##################################################################

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    
    prvs = nxt

    #temp, nb_new_labels, temp_magn, temp_dir, data, clusters = ClusterObjects(farn, struct_elem)
    
    #temp2 = np.concatenate((frame2, temp), axis=1)        
    
    im1 = ax1.imshow(frame2)
    im2 = ax2.imshow(rgb)
    ims.append([im1, im2])
        
cv2.destroyAllWindows()
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
ani.save('../../results/dense_flow.mp4', fps=25, dpi=600)
plt.show()


# -*- coding: utf-8 -*-

import SURF
import CodebookGeneration
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ISM(object):
    def __init__(self, folder, number_of_images, threshold,  descriptor_type = 'surf',
                                                             criterion = 'distance',
                                                             method = 'complete',
                                                             metric = 'correlation',
                                                             use_raw = True):
        """
        **Definition**: ISM(folder, number_of_images, threshold,  descriptor_type = 'surf', \
        criterion = 'distance', method = 'complete', metric = 'correlation', use_raw = True)
        
        Implicit Shape Model object class. Implementation of ISM creation algorithm.
        
        **Inputs**:
            * folder: path to folder containing the training images.
            * number_of_images: number of images to use for training.
            * threshold: threshold to be used during agglomerative clustering and Implicit \
            Shape Model creation.
            * descriptor_type (optional): default *surf*. Defines the descriptor to be\
            used for keypoint extraction and descriptors computation. Now, only *surf* \
            option is supported.
            * criterion (optional): default *distance*. Defines the criterion to be\
            used along threshold during agglomerative clustering.
            * method (optional): default *complete*. Defines the linkage method to be\
            used during agglomerative clustering.
            * metric (optional): default *correlation*. Defines the distance metric to be\
            used during agglomerative clustering. More information on the parameters \
            *criterion*, *method* and *metric* can be found at Scipy Hierarcical Clustering \
            documentation.            
            
        **Outputs - Class members**:
            * training_descriptors: the set of descriptors computed on entire image training \
            dataset.
            * clusters: an 1-d numpy array containing the cluster index for each observation.
            * prototypes: codebook entries (centroids) after agglomerative clustering.
            * model: Implicit Shape Model. A list cintaing the matches between codebook \
            entries and the descriptors of images of the training dataset, along with the \
            information for the candidate object center. This corresponds to the learned \
            spatial distribution.
        """
        self.folder = folder
        self.descriptor_type = descriptor_type
        self.number_of_images = number_of_images
        self.threshold = threshold
        self.criterion = criterion
        self.method = method
        self.metric = metric
        self.use_raw = True
        
        self.training_descriptors = None
        self.clusters = None
        self.prototypes = None
        self.model = None
        self.V = None
        
        
    def ComputeDescriptors(self):
        """
        **Definition**: ComputeDescriptors()
        
        Method that extract keypoints from training images and computes the descriptors for \
        these points.
        
        **Inputs**:
            * None (all necessary inputs are derived from ISM object construction)
            
        **Outputs**:
            * training_descriptors: the set of descriptors computed on entire image training \
            dataset.
        """
        print "Extracting keypoints and computing descriptors..."
        
        filename = self.folder+'0.pgm'

        img = cv2.imread(filename)
        if self.descriptor_type == 'surf':
            descriptor_object = SURF.SURF_Obj(img, upright=True)
            descriptor_object.SURF_Keypoints_Descriptors(plot_flag = False)
            
        training_descriptors = descriptor_object.descriptors
        del descriptor_object

        counter = 1
        while counter < self.number_of_images:
            filename = self.folder+'%d.pgm' % counter
            img = cv2.imread(filename)
            if self.descriptor_type == 'surf':
                descriptor_object = SURF.SURF_Obj(img, upright=False)
                descriptor_object.SURF_Keypoints_Descriptors(plot_flag = False)
            
            training_descriptors = np.concatenate((training_descriptors,
                                                   descriptor_object.descriptors), 
                                                   axis = 0)
            del descriptor_object
            counter = counter + 1
            
        self.training_descriptors = training_descriptors
        
        
    def GenerateCodebook(self):
        """
        **Definition**: GenerateCodebook()
        
        Method that computes the codebook from training images.
        
        **Inputs**:
            * None (all necessary inputs are derived from ISM object construction)
            
        **Outputs**:
            * prototypes: codebook entries (centroids) after agglomerative clustering.
            * clusters: an 1-d numpy array containing the cluster index for each observation.
        """
        print "Generating codebook..."

        code = CodebookGeneration.Codebook(self.training_descriptors, threshold = self.threshold, 
                                                                      criterion = self.criterion, 
                                                                      method = self.method, 
                                                                      metric = self.metric, 
                                                                      use_raw = self.use_raw)
        code.PrototypesExtraction()

        self.clusters = code.clusters
        self.prototypes = code.prototypes
        
        
    def GenerateModel(self):
        """
        **Definition**: GenerateModel()
        
        Method creates the Implicit Shape Model. The computed model is in the form of \
        a list.
        
        **Inputs**:
            * None (all necessary inputs are derived from ISM object construction)
            
        **Outputs**:
            * model: Implicit Shape Model. It is a list of lists. Each entry list \
            contains the matches between the codebook entries and the descriptors from \
            each training image, along with information about the candidate center of \
            the object. This corresponds to the learned spatial distribution.
        """
        print "Creating Implicit Shape Model..."

        filename = self.folder+'0.pgm'

        img = cv2.imread(filename)
        if self.descriptor_type == 'surf':
            descriptor_object = SURF.SURF_Obj(img, upright=False)
            descriptor_object.SURF_Keypoints_Descriptors(plot_flag = False)

        descriptors = descriptor_object.descriptors
        coordinates = descriptor_object.coordinates
    
        number_of_prototypes = self.prototypes.shape[0]
    
        cy = img.shape[0]/2 # the center of the image correspond to the
        cx = img.shape[1]/2 # center of the object.

        ISM_list = [] 
        for j in range(number_of_prototypes):
            prototype_list = []
            for i in range(descriptors.shape[0]):
                t = cdist(descriptors[i].reshape(1,-1), 
                          self.prototypes[j].reshape(1,-1), 
                          metric=self.metric)
                if t < self.threshold:
                    lx = coordinates[i,1]
                    ly = coordinates[i,0]
                    cx_minus_lx = cx - lx
                    cy_minus_ly = cy - ly
                    prototype_list.append(np.asarray((cy_minus_ly,cx_minus_lx)))
            
            ISM_list.append(prototype_list)
    
        del descriptor_object

        counter = 1
        while counter < self.number_of_images:
            filename = self.folder+'%d.pgm' % counter
            img = cv2.imread(filename)
            
            if self.descriptor_type == 'surf':
                descriptor_object = SURF.SURF_Obj(img, upright=False)
                descriptor_object.SURF_Keypoints_Descriptors(plot_flag = False)
    
            descriptors = descriptor_object.descriptors
            coordinates = descriptor_object.coordinates
    
            number_of_prototypes = self.prototypes.shape[0]
    
            cy = img.shape[0]/2
            cx = img.shape[1]/2

            for j in range(number_of_prototypes):
                prototype_list = []
                for i in range(descriptors.shape[0]):
                    t = cdist(descriptors[i].reshape(1,-1), 
                              self.prototypes[j].reshape(1,-1), 
                              metric=self.metric)
                    if t < self.threshold:
                        lx = coordinates[i,1]
                        ly = coordinates[i,0]
                        cx_minus_lx = cx - lx
                        cy_minus_ly = cy - ly
                        ISM_list[j].append(np.asarray((cy_minus_ly,cx_minus_lx)))
    
            del descriptor_object
            counter = counter + 1
            
        self.model = ISM_list
        
        
    def GenerateVote(self, img, plot=True):
        """
        **Definition**: GenerateVote()
        
        Method generates the Implicit Shape Model voting algorithm. The computed model is \
        in the form of a list.
        
        **Inputs**:
            * img: the new testing image in which the recognition process will be applied.
            
        **Outputs**:
            * V: Implicit Shape Model Votes. It is a list of arrays. Each entry list \
            contains the tuple (point_y, point_x, w, occ_y, occ_x, l_y, l_x).
            
                * point_y, point_x: candidate center of the object.
                * w: weigth for the vote 
                .. math:: w = p(o_n, x|C_i, l)p(C_i|f_k)
                * occ_y, occ_x: the codebook entry coordinates that match with the desctiptors \
                of the new image.
                * l_y, l_x: the coordinates of the descriptors of the new image that match with \
                the codebook entries.
        """
        print "Generating Implicit Shape Model Vote..."
        img_new = cv2.imread(img)
        if self.descriptor_type == 'surf':
            descriptor_object = SURF.SURF_Obj(img_new, upright=False)
            descriptor_object.SURF_Keypoints_Descriptors(plot_flag = False)

        descriptors = descriptor_object.descriptors
        coordinates = descriptor_object.coordinates
        threshold = self.threshold   
        coordinates_to_plot = []
              
        V = []
        for desc in range(descriptors.shape[0]):
            M = []
            for c_i in range(self.prototypes.shape[0]):
                t = cdist(descriptors[desc].reshape(1,-1), self.prototypes[c_i].reshape(1,-1), metric=self.metric)
                if t <= threshold:
                    M.append(np.asarray((c_i, coordinates[desc][0], coordinates[desc][1])))
                    coordinates_to_plot.append(np.asarray((coordinates[desc][1],coordinates[desc][0])))
            if len(M) == 0:
                continue
            p_ci_fk = 1.0/len(M)
    
            for m in range(len(M)):
                i = int(M[m][0])
                l_y = M[m][1]
                l_x = M[m][2]
        
                for occ in range(len(self.model[i])):
                    point_y = l_y - self.model[i][occ][0]
                    point_x = l_x - self.model[i][occ][1]
                    p_on_x = 1.0/len(self.model[i])
                    w = p_on_x * p_ci_fk
                    V.append(np.asarray((point_y, point_x, w, self.model[i][occ][0], self.model[i][occ][1], coordinates[desc][0], coordinates[desc][1])))
            
        self.V = V
        print "Generation of Implicit Shape Model Vote completed..."    
        
        if plot == True:
            plt.imshow(img_new, cmap = cm.Greys_r)
            for i in range(len(coordinates)):
                plt.scatter(int(coordinates_to_plot[i][0]), int(coordinates_to_plot[i][1]), s=40, c='m', marker='D')
            
            plt.show()
        
        
        
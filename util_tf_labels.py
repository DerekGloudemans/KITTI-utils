#------------------ Import necessary and unnecessary packages-----------------#
# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division

# torch and specific torch packages for convenience
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim import lr_scheduler
from torch import multiprocessing

# for convenient data loading, image representation and dataset management
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from scipy.ndimage import affine_transform

# always good to have
import time
import os
import numpy as np    
import _pickle as pickle
import random
import copy
import matplotlib.pyplot as plt
import math

from util_load import Track_Dataset, get_coords_3d

#--------------------------- Definitions section -----------------------------#
class Dataset():
    """
    Defines dataset and transforms for training or validation data. 
    """
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        #Denotes the total number of samples
        return len(self.X) 

    def __getitem__(self, index):
        x = self.X[index,:]
        y = self.Y[index,:]
        
        # convert to Tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y

def create_xy_labels(im_dir,label_dir,calib_dir):
    """
    Generates a dataset of original camera-space and transformed image-space labels
    label_dir - string, the directory of examples (for KITTI use training labels, though
                this data will be used for both training and testing)
    etc.
    
    """    
    camera_space_labels = []
    image_space_labels = []
    
    data = Track_Dataset(im_dir,label_dir,calib_dir)    
    for i in range(0, len(data.label_list)):
        
        data.load_track(i)
        labels = data.labels
        P = data.calib
        
        # each item in labels is a det_dir corresponding to one label
        for frame in labels:
            for det_dict in frame:
                if det_dict['class'] not in ["dontcare","DontCare"]:
                    # get camera space coords
                    X = det_dict['pos'][0]
                    Y = det_dict['pos'][1]
                    Z = det_dict['pos'][2]
                    h = det_dict['dim'][0]
                    w = det_dict['dim'][1]
                    l = det_dict['dim'][2]
                    alpha = det_dict['alpha']
                    rot_y = det_dict['rot_y']
                    camera_space = np.array([X,Y,Z,h,w,l,alpha,rot_y])
                    
                    #get image space coords and depths
                    tf_coords,tf_depths = get_coords_3d(det_dict,P)
                    tf_coords = tf_coords.reshape(16)
                    
                    #organize camera_space features
                    image_space = get_image_space_features(tf_coords,P)
                    
                    # remove examples behind camera
                    if X > 0 and det_dict['truncation'] == 0:
                        camera_space_labels.append(camera_space)
                        image_space_labels.append(image_space)
            
    image_space_labels = np.asarray(image_space_labels)
    camera_space_labels = np.asarray(camera_space_labels)
    return image_space_labels,camera_space_labels, tf_depths
 

def get_image_space_features(tf_coords,P):
    # flatten camera calibration matrix
    P_flat = P.reshape(12)
    
    # get a bunch of ratios in terms of image_space
    fhr =
    fwr =
    tlr = 
    blr = 
    rhr = 
    rwr = 
    
    lw  = 
    lh  = 
    
    
    
#def decode_frame_labels(label,P):
#    """
#    Gives the corresponding numpy arrays as expressed by create_datasets for a 
#    single frame
#    label - list of det_dicts
#    P - 3 x 4 camera calibration matrix
#    """
#    # each item in label is a det_dir corresponding to one detection
#    image_space_labels = []
#    camera_space_labels = []
#    for det_dict in label:
#        if det_dict['class'] not in ["dontcare","DontCare"]:
#            # get camera space coords
#            X = det_dict['pos'][0]
#            Y = det_dict['pos'][1]
#            Z = det_dict['pos'][2]
#            h = det_dict['dim'][0]
#            w = det_dict['dim'][1]
#            l = det_dict['dim'][2]
#            alpha = det_dict['alpha']
#            rot_y = det_dict['rot_y']
#            camera_space = np.array([X,Y,Z,h,w,l,alpha,rot_y])
#            
#            #get image space coords
#            tf_coords = get_coords_3d(det_dict,P).reshape([16])
#            dist = np.sqrt(X**2 + Y**2 + Z**2)
#            h_ratio = h/dist
#            w_ratio = w/dist
#            l_ratio = l/dist
#            image_space = np.array([item for item in tf_coords] + [h_ratio, w_ratio, l_ratio])
#            
#            # remove examples behind camera
#            if X > 0 :
#                camera_space_labels.append(camera_space)
#                image_space_labels.append(image_space)
#        
#    image_space_labels = np.asarray(image_space_labels)
#    camera_space_labels = np.asarray(camera_space_labels)
#    return image_space_labels,camera_space_labels        


def create_datasets(im_dir,label_dir,calib_dir, val_ratio = 0.2, seed = 0):
    """
    """
    random.seed = 0
    X,Y = create_xy_labels(im_dir,label_dir,calib_dir)
    
    n_examples = len(X)
    
    idxs = [i for i in range(0,n_examples)]
    random.shuffle(idxs)
    split_idx = int(n_examples * (1-val_ratio))
    train_idx = idxs[0:split_idx]
    test_idx = idxs[split_idx:]
    
    X_train = X[train_idx,:]
    X_test = X[test_idx,:]
    Y_train = Y[train_idx,:]
    Y_test = Y[test_idx,:]  
    
    # TODOO - better normalization
    # normalize by subtracting min value, dividing by range
    if True:
        X_min = np.min(X,0)
        Y_min = np.min(Y,0)
        X_avg = np.average(X,0)
        Y_avg = np.average(Y,0)
        X_max = np.max(X,0)
        Y_max = np.max(Y,0)
        X_train = (X_train -X_min[None,:])  /(X_avg)[None,:]
        X_test =  (X_test  -X_min[None,:])  / (X_avg)[None,:]
        Y_train = (Y_train -Y_min[None,:])  / (Y_avg)[None,:]
        Y_test =  (Y_test   -Y_min[None,:]) / (Y_avg)[None,:]
        
    train_data = Dataset(X_train,Y_train)
    test_data = Dataset(X_test,Y_test)
    
    return train_data,test_data

  
    
################################## Tester Co

if __name__ == "__main__":    
    train_im_dir =    "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
    train_lab_dir =   "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
    train_calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\data_tracking_calib(1)\\training\\calib"
    X,Y = create_datasets(train_im_dir,train_lab_dir,train_calib_dir)   

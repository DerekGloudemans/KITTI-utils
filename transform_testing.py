#------------------ Import necessary and unnecessary packages-----------------#
# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division

# torch and specific torch packages for convenience
import torch
import torch.nn.functional as F
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

from util_load import *
from util_tf_labels import *
from util_tf_net import *



if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass

    # input files and variables
    seed = 0
    random.seed = seed
    val_ratio = 0.2
    num_epochs = 300
    checkpoint_file = "checkpoints/sigmoid_3d_280.pt"
    
#    train_im_dir =    "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
#    train_lab_dir =   "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
#    train_calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\data_tracking_calib(1)\\training\\calib"
    train_im_dir =    "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Tracks/training/image_02"  
    train_lab_dir =   "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Labels/training/label_02"
    train_calib_dir = "/media/worklab/data_HDD/cv_data/KITTI/Tracking/data_tracking_calib(1)/training/calib"
    
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 0}
    
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    
    try:
        train_data
        test_data
        model
    except:
        train_data,test_data,camera_space = create_datasets(train_im_dir,train_lab_dir,train_calib_dir,val_ratio,seed) 
        trainloader = data.DataLoader(train_data, **params)
        testloader = data.DataLoader(test_data, **params)
        print("Got dataloaders.")
        
        # define FC Model
        model = FcNet()
        model = model.to(device)
        
        if checkpoint_file:
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Got model.")
    # define loss function
    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    criterion = Point_Loss()
    
    # all parameters are being optimized, not just fc layer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # define start epoch for consistent labeling if checkpoint is reloaded
    start_epoch = 0
            
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}

    
    if not checkpoint_file:   
    # train model
        print("Beginning training.")
        model = train_model(model, criterion, optimizer, 
                            exp_lr_scheduler, dataloaders,datasizes, device,
                            num_epochs, start_epoch)
        
    
    torch.cuda.empty_cache()





    #-------------------------------- test plot ----------------------------------#
    
    track_nums = [0]
    file_out =  "converted_tracks.avi".format(track_num)
    test = Track_Dataset(train_im_dir,train_lab_dir,train_calib_dir)
    
    test.load_track(track_nums[0])
    
    # load first frame
    im,label = next(test)
    
    
    # opens VideoWriter object for saving video file if necessary
    if file_out:
        writer = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('M','P','E','G'), 15.0, im.size)
    
    if False:
        for track_num in track_nums:
            test.load_track(track_num)
            
            # load first frame
            im,label = next(test)  
            
            while im:
                
                cv_im = pil_to_cv(im)
                if True:
                    cv_im = plot_bboxes_3d(cv_im,label,test.calib,style = "ground_truth")
                    
                    # try conversion
                    out = label_conversion(model,label,test.calib,im.size,device)
                    cv_im = plot_bboxes_3d(cv_im,out,test.calib)
                   
                if file_out:
                    writer.write(cv_im)
                
                cv2.imshow("Frame",cv_im)
                key = cv2.waitKey(1) & 0xff
                time.sleep(1/50.0)
                if key == ord('q'):
                    break
                
                # load next frame
                im,label = next(test)
            
        try:
            writer.release()
        except:
            pass
        cv2.destroyAllWindows()

    if True:
        # generate converted KITTI file set
        label_convert_track(test,model,out_directory = "transformed_label_test_sigmoid")
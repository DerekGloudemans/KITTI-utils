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

from util_tf_labels import *

#--------------------------- Definitions section -----------------------------#        
class FcNet(nn.Module):
    """
    Defines a simple fully connected network with 2 hidden layers
    """
    
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FcNet, self).__init__()

        # get size of some layers
        start_num = 48
        max_num = 200
        mid_num = 50
        end_num = 8
        
        # define regressor
        self.regress = nn.Sequential(
                          nn.Linear(start_num,max_num,bias=True),
                          nn.Sigmoid(),
                          nn.Linear(max_num,mid_num,bias = True),
                          nn.Sigmoid(),
                          nn.Linear(mid_num,end_num, bias = True),
                          nn.Sigmoid()
                          )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        out = self.regress(x)
        
        return out


        
    


    
def test_output(data,idx,model):
    """
    performs forward pass on Dataset item idx through model, performing necessary conversions
    """
    x,y = data[idx]
    out = model(x)
    return y.data.cpu().numpy(), out.data.cpu().numpy()
    
    
def train_model(model, criterion, optimizer, scheduler, 
                dataloaders,dataset_sizes,device, num_epochs=5, start_epoch = 0):
    """
    Alternates between a training step and a validation step at each epoch. 
    Validation results are reported but don't impact model weights
    """
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if False: #disable for Adam
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0

            # Iterate over data.
            count = 0
            for X, Y in dataloaders[phase]:
                X = X.to(device)
                Y = Y.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    Y_pred = model(X)
                

                    if type(criterion) == Point_Loss:
                        pts_2d = X[:,:16].view(-1,2,8).to(device)
                        P = (X[:,-12:].view(-1,3,4))[:,:,:3] * 1000
                        Pinv = torch.inverse(P)
                        Pinv.requires_grad = False
                        
                        # extract and invert each P matrix dumbly
                        P2 = np.array([[721.5377,0,609.5593],[0,721.5377,172.8540],[0,0,1]])
                        Pinv_test = torch.from_numpy(np.linalg.inv(P2)).float().to(device)
                        
                        loss = criterion(Y_pred,Y,Pinv,pts_2d,device)
                        
                    else:
                        loss = criterion(Y_pred,Y)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
          
                # statistics
                running_loss += loss.item()* X.size(0)
                
                # copy data to cpu and numpy arrays for scoring
                Y_pred = Y_pred.data.cpu().numpy()
                Y = Y.data.cpu().numpy()
                
                # TODO -  need some intuitive accuracy function here

    
                # verbose update
                count += 1
                if False and count % 20 == 0:
                    print("loss: {}".format(loss.item()))
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            

            print('{} Loss: {:.5f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                del best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        if epoch % 10 == 0:
            # save checkpoint
            PATH = "checkpoints/label_convert_{}.pt".format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                }, PATH)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, epoch {}'.format(best_loss,best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class Point_Loss(nn.Module):        
    def __init__(self):
        super(Point_Loss,self).__init__()
        
    def forward(self,output,target,Pinv,pts_2d,device):
        """ Assumes Pinv and pts_2d are tensors"""
        
        # concatenate a third row of ones with pts_2d
        ones = torch.ones((pts_2d.shape[0],1,pts_2d.shape[-1])).to(device)
        pts_2d = torch.cat((pts_2d,ones),1)
        
        # multiply pts_2d by output and target, respectively
        output = output.unsqueeze(1)
        target = target.unsqueeze(1)
        out_scaled = torch.mul(pts_2d,output)
        targ_scaled = torch.mul(pts_2d,target)
    
        # multiply each result by Pinv
        out_3d = torch.bmm(Pinv,out_scaled)
        targ_3d = torch.bmm(Pinv,targ_scaled)
        loss = nn.SmoothL1Loss()#MSELoss()
        return loss(out_3d,targ_3d)

   
        
#------------------------------ Main code here -------------------------------#
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')    
    except:
        pass
    
    # use this to watch gpu in console            watch -n 2 nvidia-smi
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()


    # input files and variables
    seed = 0
    random.seed = seed
    val_ratio = 0.2
    num_epochs = 250
    checkpoint_file = "pts_L1_140.pt"
    train_im_dir =    "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
    train_lab_dir =   "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
    train_calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\data_tracking_calib(1)\\training\\calib"
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 0}
    
    
    
    train_data,test_data,camera_space = create_datasets(train_im_dir,train_lab_dir,train_calib_dir,val_ratio,seed) 
    trainloader = data.DataLoader(train_data, **params)
    testloader = data.DataLoader(test_data, **params)
    print("Got dataloaders.")
    

    # define FC Model
    model = FcNet()
    model = model.to(device)
    print("Got model.")
    
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
    
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

#    # if checkpoint specified, load model and optimizer weights from checkpoint
#    if checkpoint_file != None:
#        model,optimizer,start_epoch = load_model(checkpoint_file, model, optimizer)
#        #model,_,start_epoch = load_model(checkpoint_file, model, optimizer) # optimizer restarts from scratch
#        print("Checkpoint loaded.")
            
    # group dataloaders
    dataloaders = {"train":trainloader, "val": testloader}
    datasizes = {"train": len(train_data), "val": len(test_data)}
#    dataloaders = {"val":trainloader, "train": testloader}
#    datasizes = {"val": len(train_data), "train": len(test_data)}
    
    if False:   
    # train model
        print("Beginning training.")
        model = train_model(model, criterion, optimizer, 
                            exp_lr_scheduler, dataloaders,datasizes, device,
                            num_epochs, start_epoch)
        
    
    torch.cuda.empty_cache()
    
    
      


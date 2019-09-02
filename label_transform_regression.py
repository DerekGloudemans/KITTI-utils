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
        start_num = 19
        max_num = 38
        mid_num = 16
        end_num = 8
        
        # define regressor
        self.regress = nn.Sequential(
                          nn.Linear(start_num,max_num,bias=True),
                          nn.ReLU(),
                          nn.Linear(max_num,mid_num,bias = True),
                          nn.ReLU(),
                          nn.Linear(mid_num,end_num, bias = True),
                          nn.ReLU()
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
    return y, out.data.cpu().numpy()
    
    
    
def train_model(model, criterion, optimizer, scheduler, 
                dataloaders,dataset_sizes, num_epochs=5, start_epoch = 0):
    """
    Alternates between a training step and a validation step at each epoch. 
    Validation results are reported but don't impact model weights
    """
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
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
                
                    # note that the classification loss is done using class-wise probs rather 
                    # than a single class label?
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
                if count % 20 == 0:
                    print("loss: {}".format(loss.item()))
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                del best_model_wts
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        
        if epoch % 3 == 0:
            # save checkpoint
            PATH = "splitnet_centered_checkpoint_{}.pt".format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                }, PATH)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

        
        
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
    random.seed = 0
    val_ratio = 0.2
    x_path = "C:\\Users\\derek\\OneDrive\\Documents\\Derek's stuff\\Not Not School\\Lab\\Code\\KITTI-utils\\label_dataset\\X.npy"
    y_path = "C:\\Users\\derek\\OneDrive\\Documents\\Derek's stuff\\Not Not School\\Lab\\Code\\KITTI-utils\\label_dataset\\Y.npy"
    checkpoint_file = None
    
    
    # split data into train and test data
    X = np.load(x_path)
    Y = np.load(y_path)
    n_examples = len(X)
    idxs = [i for i in range(0,n_examples)]
    random.shuffle(idxs)
    split_idx = int(n_examples * (1-val_ratio))
    X_train = X[idxs[0:split_idx]]
    X_test = X[idxs[split_idx:]]
    Y_train = Y[idxs[0:split_idx]]
    Y_test = Y[idxs[split_idx:]]  
    
    # normalize by max training set values
    if False:
        X_norm = np.max(X,0)
        Y_norm = np.max(Y,0)
        X_train = X_train / X_norm[None,:]
        X_test = X_test / X_norm[None,:]
        Y_train = Y_train / Y_norm[None,:]
        Y_test = Y_test / Y_norm[None,:]
    
    
    # TODO = fix this create training params
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 0}
    num_epochs = 10
    
    
    
   
    train_data = Dataset(X_train,Y_train)
    test_data = Dataset(X_test,Y_test)
    trainloader = data.DataLoader(train_data, **params)
    testloader = data.DataLoader(test_data, **params)
    print("Got dataloaders.")
    

    # define FC Model
    model = FcNet()
    model = model.to(device)
    print("Got model.")
    
    # define loss function
    criterion = nn.MSELoss()

    # all parameters are being optimized, not just fc layer
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
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
    dataloaders = {"val":trainloader, "train": testloader}
    datasizes = {"val": len(train_data), "train": len(test_data)}
#    dataloaders = {"val":trainloader, "train": testloader}
#    datasizes = {"val": len(train_data), "train": len(test_data)}
    
    if True:   
    # train model
        print("Beginning training.")
        model = train_model(model, criterion, optimizer, 
                            exp_lr_scheduler, dataloaders,datasizes,
                            num_epochs, start_epoch)
        
    
    torch.cuda.empty_cache()

    correct, pred = test_output(train_data,10,model) 
    out2 = out * Y_norm
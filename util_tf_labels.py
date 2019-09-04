"""
Contains functions for converting from TrackDataset format to tensor format for 
training a label conversion network.
"""


#------------------ Import necessary and unnecessary packages-----------------#
# this seems to be a popular thing to do so I've done it here
from __future__ import print_function, division

# torch and specific torch packages for convenience
import torch

# always good to have
import numpy as np    
import random

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
    depth_labels = []
    data = Track_Dataset(im_dir,label_dir,calib_dir)    
    for i in range(0, len(data.label_list)):
        
        data.load_track(i)
        im,_ = next(data)
        im_size = im.size[:2]
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
                    image_space = get_image_space_features(tf_coords,P,im_size)
                    
                    # remove examples behind camera
                    if X > 0 and det_dict['truncation'] == 0:
                        camera_space_labels.append(camera_space)
                        image_space_labels.append(image_space)
                        depth_labels.append(tf_depths)
            
            
            
    image_space_labels = np.asarray(image_space_labels)
    camera_space_labels = np.asarray(camera_space_labels)
    depth_labels = np.asarray(depth_labels) /100 # to normalize
     
    return image_space_labels,depth_labels, camera_space_labels 
 

def get_image_space_features(tf_coords,P,im_size):
    """
    converts xy coordinates for 8 bbox corners into normalized featurespace for training
    tf_coords - 2 x 8 numpy array
    P - 3 x 4 numpy array, camera calibration matrix
    im_size - 2-tuple, image dimensions
    """
    im_size = np.array([im_size[1],im_size[0]])[None,:] # add second axis
    
    if np.shape(tf_coords) == (2,8):
        tf_coords = tf_coords.reshape(16)
    
    # flatten camera calibration matrix
    P_flat = P.reshape(12)
    c = tf_coords # shorten name for expressions below
    dist = lambda x1,y1,x2,y2: np.sqrt((x1-x2)**2 + (y1-y2)**2)
    # get a bunch of ratios in terms of image_space
    fhr = dist(c[0],c[8],c[4],c[12])  / dist(c[1],c[9],c[5],c[13])
    fwr = dist(c[0],c[8],c[1],c[9])   / dist(c[4],c[12],c[5],c[13])
    tlr = dist(c[0],c[8],c[3],c[11])  / dist(c[1],c[9],c[2],c[10])
    blr = dist(c[4],c[12],c[7],c[15]) / dist(c[5],c[13],c[6],c[14])
    rhr = dist(c[3],c[11],c[7],c[15]) / dist(c[2],c[10],c[6],c[14])
    rwr = dist(c[3],c[11],c[2],c[10]) / dist(c[7],c[15],c[6],c[14])
    
    lw  = (dist(c[0],c[8],c[3],c[11]) + dist(c[1],c[9],c[2],c[10])) / \
    (dist(c[3],c[11],c[2],c[10]) + dist(c[0],c[8],c[1],c[9]) )
    
    lh  = (dist(c[0],c[8],c[3],c[11]) + dist(c[1],c[9],c[2],c[10])) / \
    (dist(c[3],c[11],c[7],c[15]) + dist(c[2],c[10],c[6],c[14]))
    
    ratios = [fhr,fwr,tlr,blr,rhr,rwr,lw,lh]
    
    # normalize ratios by dividing by 10
    ratios = [j/2 for j in ratios]
    # normalize P by dividing by 1000
    P_flat = [k/1000 for k in P_flat]
    # normalize coords by dividing by image size
    tf_coords = tf_coords.reshape(8,2) / im_size
    tf_coords = tf_coords.reshape(16)
    ret = np.array([i for i in tf_coords] + [j for j in ratios] + [k for k in P_flat])
    return ret
    
    
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
    creates datasets of tensors for PyTorch training
    label_dir - string, the directory of examples (for KITTI use training labels, though
                this data will be used for both training and testing)
    calib_dir - "
    im_dir - "
    val_ratio - ratio of data to put in test set
    seed - integer, random seed for repeatability
    """
    random.seed = seed
    X,Y,camera_space = create_xy_labels(im_dir,label_dir,calib_dir)
    
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
        
    train_data = Dataset(X_train,Y_train)
    test_data = Dataset(X_test,Y_test)
    
    return train_data,test_data,camera_space

def im_to_cam_space(pts_2d,pts_depth,P):
    """
    Converts a set of image_space points into a set of camera-space points
    pts_2d -  2 x 8 numpy array of xy coordinates for each corner
    pts_depth - 1 x 8 numpy array with depth in z direction (predicted by network)
    P - 3 x 4 camera calibration matrix
    """
    
    # scale pts_2d by pts_depth
    pts_2d[0,:] = np.multiply(pts_2d[0,:],pts_depth)
    pts_2d[1,:] = np.multiply(pts_2d[1,:],pts_depth)

    
    P_inv = np.linalg.inv(P[:,0:3])
    pts_2d = np.concatenate((pts_2d,pts_depth),0)
    
    
    
    pts_3d = np.matmul(P_inv,pts_2d)
    return pts_3d

def label_conversion(model,label,P,im_size):
    """ Takes in one label (bboxes for one frame). For each, converts into 
        image space, uses network to predict camera_space, and averages out points 
        to create best fit 3d bounding box. Returns a list of det_dict objects
        label - list of det_dicts
        P - 3 x 4 numpy array, camera calibration matrix
        im_size - 2-tuple, size of image
    """
    new_label = []
    
    for det_dict in label:
        if det_dict['class'] not in ['DontCare', 'dontcare']:
            # get image coords
            coords, real_depth, real_3d = get_coords_3d(det_dict,P)
            X = get_image_space_features(coords,P,im_size)
            X = torch.from_numpy(X).float()
            
            # model output depths
            pred_depths = (model(X).data.cpu().numpy())[None,:]*100
            
            # convert into camera space again
            pts_3d = im_to_cam_space(coords,real_depth[None,:],P)
            
            X = np.average(pts_3d[0])
            Y = np.average(pts_3d[1])
            Z = np.average(pts_3d[2])
            
            # find best l,w,h 
            dist = lambda pts,a,b: np.sqrt((pts[0,a]-pts[0,b])**2 + \
                                           (pts[1,a]-pts[1,b])**2 + \
                                           (pts[2,a]-pts[2,b])**2)
            
            length  = (dist(pts_3d,0,3) + dist(pts_3d,1,2) + \
                      dist(pts_3d,4,7) + dist(pts_3d,5,6)) /4.0
            width  =  (dist(pts_3d,0,1) + dist(pts_3d,3,2) + \
                      dist(pts_3d,4,5) + dist(pts_3d,7,6)) /4.0
            height =  (dist(pts_3d,0,4) + dist(pts_3d,1,5) + \
                      dist(pts_3d,2,6) + dist(pts_3d,3,7)) /4.0
            
            # find best alpha by averaging angles of all 8 relevant line segments
            # defined for line segments backwards to forwards and left to right
            ang = lambda pts,a,b: np.arctan((pts[1,b]-pts[1,a])/(pts[0,b]-pts[0,a]))
            
            angle = (ang(pts_3d,0,1) + ang(pts_3d,3,2) + ang(pts_3d,4,5) + ang(pts_3d,7,6))/4.0 + \
                    ((ang(pts_3d,3,0) + ang(pts_3d,2,1) + ang(pts_3d,7,4) + ang(pts_3d,6,5))/4.0 - np.pi/2)
            alpha = (np.pi - angle)
    
            # append to new label
            det_dict['pos'][0] = X
            det_dict['pos'][1] = Y
            det_dict['pos'][2] = Z
            det_dict['dim'][0] = height
            det_dict['dim'][1] = width
            det_dict['dim'][2] = length
            det_dict['alpha'] = alpha
            
            new_label.append(det_dict)
        
    return new_label
        
    
    
#---------------------------------Tester Code---------------------------------#

if __name__ == "__main__":    
    train_im_dir =    "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
    train_lab_dir =   "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
    train_calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\data_tracking_calib(1)\\training\\calib"
    train_data,test_data,camera_space = create_datasets(train_im_dir,train_lab_dir,train_calib_dir)   

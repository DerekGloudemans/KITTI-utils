"""
Parse KITTI 3D object dataset into separate images for each object in each frame
and get associated labels
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import _pickle as pickle

import cv2
import PIL
from PIL import Image
from math import cos,sin
from util_load import get_coords_3d, plot_text, pil_to_cv

def parse_label_file2(label_list,idx):
    """parse label text file into a list of numpy arrays, one for each frame"""
    f = open(label_list[idx])
    line_list = []
    for line in f:
        line = line.split()
        line_list.append(line)
        
    # each line corresponds to one detection
    det_dict_list = []  
    for line in line_list:
        # det_dict holds info on one detection
        det_dict = {}
        det_dict['class']      = str(line[0])
        det_dict['truncation'] = float(line[1])
        det_dict['occlusion']  = int(line[2])
        det_dict['alpha']      = float(line[3]) # obs angle relative to straight in front of camera
        x_min = int(round(float(line[4])))
        y_min = int(round(float(line[5])))
        x_max = int(round(float(line[6])))
        y_max = int(round(float(line[7])))
        det_dict['bbox2d']     = np.array([x_min,y_min,x_max,y_max])
        length = float(line[10])
        width = float(line[9])
        height = float(line[8])
        det_dict['dim'] = np.array([length,width,height])
        x_pos = float(line[11])
        y_pos = float(line[12])
        z_pos = float(line[13])
        det_dict['pos'] = np.array([x_pos,y_pos,z_pos])
        det_dict['rot_y'] = float(line[14])
        det_dict_list.append(det_dict)
    
    return det_dict_list


def parse_calib_file(calib_list,idx):
    """parse calib file to get  camera projection matrix"""
    f = open(calib_list[idx])
    line_list = []
    for line in f:
        line = line.split()
        line_list.append(line)
    line = line_list[2] # get line corresponding to left color camera
    vals = np.zeros([12])
    for i in range(0,12):
        vals[i] = float(line[i+1])
    calib = vals.reshape((3,4))
    return calib

    
def plot_bbox_2d2(im,det):
    """ Plots rectangular bbox on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one det_dict, in the form output by parse_label_file 
    bbox_im -  cv2 im with bboxes and labels plotted
    """
    
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.Image.Image,PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    
    if type(im) in [PIL.PngImagePlugin.PngImageFile,PIL.Image.Image]:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(255,100,0),
            'Person':(255,50,0),
            'Car': (0,255,150),
            'Van': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    bbox = det['bbox2d']
    cls = det['class']
    
    cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[cls], 1)
    if cls != 'DontCare':
        plot_text(cv_im,(bbox[0],bbox[1]),cls,0,class_colors)
    return cv_im 

 
    
##############################################################################    



image_dir = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object\\Images\\training\\image_2"
label_dir = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object\\Detection_Labels\\data_object_label_2\\training\\label_2"
calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object\\data_object_calib\\training\\calib"
new_dir = "C:\\Users\\derek\\Desktop\\KITTI\\3D_object_parsed"
buffer = 25

# create new directory for holding image crops
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    os.mkdir(os.path.join(new_dir,"images"))

# stores files for each set of images and each label
dir_list = next(os.walk(image_dir))[1]
image_list = [os.path.join(image_dir,item) for item in os.listdir(image_dir)]
label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
calib_list = [os.path.join(calib_dir,item) for item in os.listdir(calib_dir)]
image_list.sort()
label_list.sort()
calib_list.sort()


out_labels = {}
# loop through images
for i in range(len(image_list)):
    if i %50 == 0:
        print("On image {} of {}".format(i,len(image_list)))
        
    # get label and cilbration matrix
    det_dict_list = parse_label_file2(label_list,i)
    calib = parse_calib_file(calib_list,i)
    
    # open image
    im = Image.open(image_list[i])
    imsize = im.size
    if False:
        im.show()
    
    # loop through objects in frame
    obj_count = 0
    for j in range(len(det_dict_list)):
        det = det_dict_list[j]
        if det['class'] not in ["dontcare", "DontCare"]:
            crop = np.zeros(4)
            crop[0] = max(det['bbox2d'][0]-buffer,0) #xmin, left
            crop[1] = max(det['bbox2d'][1]-buffer,0) #ymin, top
            crop[2] = min(det['bbox2d'][2]+buffer,imsize[0]-1) #xmax
            crop[3] = min(det['bbox2d'][3]+buffer,imsize[1]-1) #ymax
            det['offset'] = (crop[0],crop[1])    
    
            
            crop_im = im.crop(crop)
            
            det['bbox2d'][0] = det['bbox2d'][0] - crop[0]
            det['bbox2d'][1] = det['bbox2d'][1] - crop[1]
            det['bbox2d'][2] = det['bbox2d'][2] - crop[0]
            det['bbox2d'][3] = det['bbox2d'][3] - crop[1]
            
            # show crop if true
            if False:
                box_im = plot_bbox_2d2(crop_im,det)
                cv2.imshow("Frame",box_im)
                key = cv2.waitKey(0) & 0xff
                #time.sleep(1/30.0)
                if key == ord('q'):
                    break
            
            # save image
            im_name = "{:06d}-{:02d}.png".format(i,obj_count)
            crop_im.save(os.path.join(new_dir,"images",im_name))
            obj_count += 1
        
            # create label - add calib and bbox3d to det_dict
            det['calib'] = calib
            det['bbox3d'],det['depths'],det['cam_space'] = get_coords_3d(det,calib)
            
            # replace old det_dict with det in det_dict_list
            out_labels[im_name] = det
    
# pickle out_labels
with open(os.path.join(new_dir,"labels.cpkl"),'wb') as f:
    pickle.dump(out_labels,f)
print("Done parsing 3D detection dataset.")

with open(os.path.join(new_dir,"labels.cpkl"),'rb') as f:
    loaded_labels = pickle.load(f)
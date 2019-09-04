""" 
This file contains utilities for loading and showing tracks from the KITTI dataset 
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np

import cv2
import PIL
from PIL import Image
from math import cos,sin

class Track_Dataset():
    """
    Creates an object for referencing the KITTI object tracking dataset (training set)
    """
    
    def __init__(self, image_dir, label_dir,calib_dir):
        """ initializes object. By default, the first track (cur_track = 0) is loaded 
        such that next(object) will pull next frame from the first track"""

        # stores files for each set of images and each label
        dir_list = next(os.walk(image_dir))[1]
        self.track_list = [os.path.join(image_dir,item) for item in dir_list]
        self.label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
        self.calib_list = [os.path.join(calib_dir,item) for item in os.listdir(calib_dir)]
        self.track_list.sort()
        self.label_list.sort()
        self.calib_list.sort()
        
        # keep track of current track (sequence of frames)
        # these variables are assigned new values in the load_track operation below
        # and are listed here only for convenience
        self.cur_track =  None
        self.cur_track_path = None
        self.labels = None
        self.calib = None
        # keep track of current frame
        self.cur_frame = None
        
        # load track 0
        self.load_track(0)
        
        
    def load_track(self,idx):
        """moves to track indexed"""
        self.cur_track = idx
        self.cur_track_path = self.track_list[idx]
        self.cur_frame = -1 # so that calling next will load frame 0
        self.parse_label_file(idx)
        self.parse_calib_file(idx)
        
    def __len__(self):
        """ return number of tracks"""
        return len(self.track_list)
    
    def track_len(self):
        """ return number of frames of current track """
        return len(self.labels)
    
    def __next__(self):
        """get next frame and label from current track"""
        self.cur_frame = self.cur_frame + 1
        try:
            im = Image.open(os.path.join(self.cur_track_path,"{:06d}.png".format(self.cur_frame)))
            labels = self.labels[self.cur_frame]
            return im, labels
        
        except FileNotFoundError:
            print("End of track.")
            return None, None
    
    def parse_label_file(self,idx):
        """parse label text file into a list of numpy arrays, one for each frame"""
        f = open(self.label_list[idx])
        line_list = []
        for line in f:
            line = line.split()
            line_list.append(line)
            
        # each line corresponds to one detection
        det_dict_list = []  
        for line in line_list:
            # det_dict holds info on one detection
            det_dict = {}
            det_dict['frame']      = int(line[0])
            det_dict['id']         = int(line[1])
            det_dict['class']      = str(line[2])
            det_dict['truncation'] = int(line[3])
            det_dict['occlusion']  = int(line[4])
            det_dict['alpha']      = float(line[5]) # obs angle relative to straight in front of camera
            x_min = int(round(float(line[6])))
            y_min = int(round(float(line[7])))
            x_max = int(round(float(line[8])))
            y_max = int(round(float(line[9])))
            det_dict['bbox2d']     = np.array([x_min,y_min,x_max,y_max])
            length = float(line[12])
            width = float(line[11])
            height = float(line[10])
            det_dict['dim'] = np.array([length,width,height])
            x_pos = float(line[13])
            y_pos = float(line[14])
            z_pos = float(line[15])
            det_dict['pos'] = np.array([x_pos,y_pos,z_pos])
            det_dict['rot_y'] = float(line[16])
            det_dict_list.append(det_dict)
        
        # pack all detections for a frame into one list
        label_list = []
        idx = 0
        frame_det_list = []
        
        for det in det_dict_list:
            assigned = False
            while assigned == False:
                if det['frame'] == idx:
                    frame_det_list.append(det)
                    assigned = True
                # no more detections from frame idx, advance to next
                else:
                    label_list.append(frame_det_list)
                    frame_det_list = []
                    idx = idx + 1
        label_list.append(frame_det_list) # append last frame detections
        self.labels = label_list
    
    def parse_calib_file(self,idx):
        """parse calib file to get  camera projection matrix"""
        f = open(self.calib_list[idx])
        line_list = []
        for line in f:
            line = line.split()
            line_list.append(line)
        line = line_list[2] # get line corresponding to left color camera
        vals = np.zeros([12])
        for i in range(0,12):
            vals[i] = float(line[i+1])
        self.calib = vals.reshape((3,4))
        
        
    # End Track_Dataset definition
 
     
def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def plot_text(im,offset,cls,idnum,class_colors):
    """ Plots filled text box on original image, 
        utility function for plot_bboxes_2d """
    
    text = "{}: {}".format(idnum,cls)
    
    font_scale = 1.0
    font = cv2.FONT_HERSHEY_PLAIN
    
    # set the rectangle background to white
    rectangle_bgr = class_colors[cls]
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    
    # set the text start position
    text_offset_x = int(offset[0])
    text_offset_y = int(offset[1])
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)


def plot_bboxes_2d(im,label):
    """ Plots rectangular bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file 
    bbox_im -  cv2 im with bboxes and labels plotted
    """
    
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(im) == PIL.PngImagePlugin.PngImageFile:
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
    
    for det in label:
        bbox = det['bbox2d']
        cls = det['class']
        idnum = det['id']
        
        cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[cls], 1)
        if cls != 'DontCare':
            plot_text(cv_im,(bbox[0],bbox[1]),cls,idnum,class_colors)
    return cv_im


def get_coords_3d(det_dict,P):
    """ returns the pixel-space coordinates of an object's 3d bounding box
        computed from the label and the camera parameters matrix
        for the idx object in the current frame
        det_dict - object representing one detection
        P - camera calibration matrix
        bbox3d - 8x2 numpy array with x,y coords for ________ """     
    # create matrix of bbox coords in physical space 

    l = det_dict['dim'][0]
    w = det_dict['dim'][1]
    h = det_dict['dim'][2]
    x_pos = det_dict['pos'][0]
    y_pos = det_dict['pos'][1]
    z_pos = det_dict['pos'][2]
    ry = det_dict['rot_y']
    cls = det_dict['class']
        
        
    # in absolute space (meters relative to obj center)
    obj_coord_array = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],
                                [0,0,0,0,-h,-h,-h,-h],
                                [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]])
    
    # apply object-centered rotation here
    R = np.array([[cos(ry),0,sin(ry)],[0,1,0],[-sin(ry),0,cos(ry)]])
    rotated_corners = np.matmul(R,obj_coord_array)
    
    rotated_corners[0,:] += x_pos
    rotated_corners[1,:] += y_pos
    rotated_corners[2,:] += z_pos
    
    # transform with calibration matrix
    # add 4th row for matrix multiplication
    zeros = np.zeros([1,np.size(rotated_corners,1)])
    rotated_corners = np.concatenate((rotated_corners,zeros),0)

    
    pts_2d = np.matmul(P,rotated_corners)
    pts_2d[0,:] = pts_2d[0,:] / pts_2d[2,:]        
    pts_2d[1,:] = pts_2d[1,:] / pts_2d[2,:] 
    
    # apply camera space rotation here?
    return pts_2d[:2,:] ,pts_2d[2,:]

    
def draw_prism(im,coords,color):
    """ draws a rectangular prism on a copy of an image given the x,y coordinates 
    of the 8 corner points, does not make a copy of original image
    im - cv2 image
    coords - 2x8 numpy array with x,y coords for each corner
    prism_im - cv2 image with prism drawn"""
    prism_im = im.copy()
    coords = np.transpose(coords).astype(int)
    #fbr,fbl,rbl,rbr,ftr,ftl,frl,frr
    edge_array= np.array([[0,1,0,1,1,0,0,0],
                          [1,0,1,0,0,1,0,0],
                          [0,1,0,1,0,0,1,1],
                          [1,0,1,0,0,0,1,1],
                          [1,0,0,0,0,1,0,1],
                          [0,1,0,0,1,0,1,0],
                          [0,0,1,0,0,1,0,1],
                          [0,0,0,1,1,0,1,0]])

    # plot lines between indicated corner points
    for i in range(0,8):
        for j in range(0,8):
            if edge_array[i,j] == 1:
                cv2.line(prism_im,(coords[i,0],coords[i,1]),(coords[j,0],coords[j,1]),color,1)
    return prism_im


def plot_bboxes_3d(im,label,P):
    """ Plots rectangular prism bboxes on image and returns image
    im - cv2 or PIL style image (function converts to cv2 style image)
    label - for one frame, in the form output by parse_label_file
    P - camera calibration matrix
    bbox_im -  cv2 im with bboxes and labels plotted
    """
        
    # check type and convert PIL im to cv2 im if necessary
    assert type(im) in [np.ndarray, PIL.PngImagePlugin.PngImageFile], "Invalid image format"
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        im = pil_to_cv(im)
    cv_im = im.copy() 
    
    class_colors = {
            'Cyclist': (255,150,0),
            'Pedestrian':(200,800,0),
            'Person':(160,30,0),
            'Car': (0,255,150),
            'Van': (0,255,100),
            'Truck': (0,255,50),
            'Tram': (0,100,255),
            'Misc': (0,50,255),
            'DontCare': (200,200,200)}
    
    for i in range (0,len(label)):
        
        cls = label[i]['class']
        idnum = label[i]['id']
        if cls != "DontCare":
            bbox_3d,_ = get_coords_3d(label[i],P)
            cv_im = draw_prism(cv_im,bbox_3d,class_colors[cls])
            plot_text(cv_im,(bbox_3d[0,4],bbox_3d[1,4]),cls,idnum,class_colors)
    return cv_im

 ################################# Tester Code ################################    
if __name__ == "__main__":    
    train_im_dir =    "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
    train_lab_dir =   "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
    train_calib_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\data_tracking_calib(1)\\training\\calib"
    
    #train_im_dir =    "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Tracks/training/image_02"  
    #train_lab_dir =   "/media/worklab/data_HDD/cv_data/KITTI/Tracking/Labels/training/label_02"
    #train_calib_dir = "/media/worklab/data_HDD/cv_data/KITTI/Tracking/data_tracking_calib(1)/training/calib"
    
    test = Track_Dataset(train_im_dir,train_lab_dir,train_calib_dir)
    test.load_track(3)
    
    
    
    im,label = next(test)
    
    while im:
        
        cv_im = pil_to_cv(im)
        if True:
            cv_im = plot_bboxes_3d(cv_im,label,test.calib)
        cv2.imshow("Frame",cv_im)
        key = cv2.waitKey(1) & 0xff
        #time.sleep(1/30.0)
        if key == ord('q'):
            break
        
        # load next frame
        im,label = next(test)
    
        
    cv2.destroyAllWindows()
    
   
    
    
            
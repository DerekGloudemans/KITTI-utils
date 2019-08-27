import os
import time
import matplotlib.pyplot as plt
import numpy as np

import cv2
import PIL
from PIL import Image

class Track_Dataset():
    """
    Creates an object for referencing the KITTI object tracking dataset (training set)
    """
    
    def __init__(self, image_dir, label_dir):
        """ initializes object. By default, the first track (cur_track = 0) is loaded 
        such that next(object) will pull next frame from the first track"""

        # stores files for each set of images and each label
        dir_list = next(os.walk(image_dir))[1]
        self.track_list = [os.path.join(image_dir,item) for item in dir_list]
        self.label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
        
        # keep track of current track (sequence of frames)
        # these variables are assigned new values in the load_track operation below
        # and are listed here only for convenience
        self.cur_track = 0
        self.cur_track_path = None
        self.labels = []
        
        # keep track of current frame
        self.cur_frame = -1
        
        # load track 0
        self.load_track(0)
        
        
    def load_track(self,idx):
        """moves to track indexed"""
        self.cur_track = idx
        self.cur_track_path = self.track_list[idx]
        self.cur_frame = -1 # so that calling next will load frame 0
        self.labels = self.parse_label_file(self.label_list[idx])
    
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
    
    def parse_label_file(self,file):
        "parse label text file into a list of numpy arrays, one for each frame"
        f = open(file)
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
            x_pos = float(line[10])
            y_pos = float(line[11])
            z_pos = float(line[12])
            det_dict['pos'] = np.array([x_pos,y_pos,z_pos])
            det_dict['rot_y'] = float(line[13])
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
        return label_list

def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 

def plot_text(im,bbox,cls,idnum,class_colors):
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
    text_offset_x = bbox[0]
    text_offset_y = bbox[1]
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
            'PersonSitting':(255,50,0),
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
            plot_text(cv_im,bbox,cls,idnum,class_colors)
    return cv_im
    
############################################## start  tester code here    
    
train_im_dir =  "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
train_lab_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
test = Track_Dataset(train_im_dir,train_lab_dir)
test.load_track(10)

im,label = next(test)

while im:
    
    cv_im = pil_to_cv(im)
    if True:
        cv_im = plot_bboxes_2d(im,label)
    cv2.imshow("Frame",cv_im)
    key = cv2.waitKey(1) & 0xff
    time.sleep(1/30.0)
    
    if key == ord('q'):
        break
    
    # load next frame
    im,label = next(test)

    
cv2.destroyAllWindows()




        
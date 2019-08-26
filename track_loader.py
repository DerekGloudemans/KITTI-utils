import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Track_Dataset():
    """
    Creates an object for referencing the KITTI object tracking dataset
    """
    
    def __init__(self, image_dir, label_dir):
        self.im_dir = image_dir
        self.lab_dir = label_dir
        
        dir_list = next(os.walk(image_dir))[1]
        self.track_list = [os.path.join(image_dir,item) for item in dir_list]
        self.label_list = [os.path.join(image_dir,item) for item in os.listdir(label_dir)]
        
        self.cur_track = 0
        self.cur_track_path = ""
        self.cur_frame = 0
        self.label_arr_list = [0 for i in range(10000)]
        
        self.load_track(0)
        
    def load_track(self,idx):
        "moves to track indexed"
        self.cur_track = idx
        self.cur_track_path = self.track_list[idx]
        self.cur_frame = -1
        
    
    def __next__(self):
        "get next frame and label from current track"
        self.cur_frame = self.cur_frame + 1
        im = Image.open(os.path.join(self.cur_track_path,"{:06d}.png".format(self.cur_frame)))
        labels = self.label_arr_list[self.cur_frame]
        return im, labels
    
    def parse_label_file(self):
        pass
    
train_im_dir =  "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Tracks\\training\\image_02"  
train_lab_dir = "C:\\Users\\derek\\Desktop\\KITTI\\Tracking\\Labels\\training\\label_02"
test = Track_Dataset(train_im_dir,train_lab_dir)

im,junk_label = next(test)
plt.imshow(im)
from __future__ import division
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import time
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random
from filterpy.kalman import KalmanFilter
from util_draw import draw_track

def condense_detections(detections,style = "center"):
    """
    converts the input data object (from the yolo detector or similar) into the 
    specified output style
    
    detections - input list of length equal to number of frames. Each list item is
    a D x 8 numpy array with a row for each object containing:
    index of the image in the batch (always 0 in this implementation 
    4 corner coordinates (x1,y1,x2,y2), objectness score, the score of class 
    with maximum confidence, and the index of that class.
    
    will return a list of D x ? numpy arrays with the contents of each row as 
    specified by style parameter
    
    style "center" -  centroid x, centroid y
    style "bottom_center" - centroid x, bottom y
    style "SORT" - centroid x, centroid y, scale (height) and ratio (width/height)
    style "SORT_with_conf" - as above plus detection confidence
    """
    assert style in ["SORT_with_class","SORT","center","bottom_center"], "Invalid style input."
    
    new_list = []
    
    if style == "center":
        for item in detections:
            coords = np.zeros([len(item),2])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0
                coords[i,1] = (item[i,2]+item[i,4])/2.0
            new_list.append(coords) 
            
    elif style == "bottom_center":   
        for item in detections:
            coords = np.zeros([len(item),2])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0
                coords[i,1] = item[i,4]
            new_list.append(coords)
            
    elif style == "SORT":
        for item in detections:
            coords = np.zeros([len(item),4])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0 # x centroid
                coords[i,1] = (item[i,2]+item[i,4])/2.0 # y centroid
                coords[i,2] = (item[i,4]-item[i,2]) # scale (y height)
                coords[i,3] = (item[i,3]-item[i,1])/float(coords[i,2]) # ratio (width/height)
            new_list.append(coords)
            
    elif style == "SORT_with_class":
        for item in detections:
            coords = np.zeros([len(item),6])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0 # x centroid
                coords[i,1] = (item[i,2]+item[i,4])/2.0 # y centroid
                coords[i,2] = (item[i,4]-item[i,2]) # scale (y height)
                coords[i,3] = (item[i,3]-item[i,1])/float(coords[i,2]) # ratio (width/height)
                coords[i,4] = (item[i,6]) # class confidence
                coords[i,5] = (item[i,7]) # class label
            new_list.append(coords)
       
    return new_list


def match_greedy(first,second,threshold = 10):
    """
    performs  greedy best-first matching of objects between frames
    inputs - N x 2 arrays of object x and y coordinates from different frames
    output - M x 1 array where index i corresponds to the second frame object 
    matched to the first frame object i
    """

    # find distances between first and second
    dist = np.zeros([len(first),len(second)])
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
    
    # select closest pair
    matchings = np.zeros(len(first))-1
    unflat = lambda x: (x//len(second), x %len(second))
    while np.min(dist) < threshold:
        min_f, min_s = unflat(np.argmin(dist))
        #print(min_f,min_s,len(first),len(second),len(matchings),np.argmin(dist))
        matchings[min_f] = min_s
        dist[:,min_s] = np.inf
        dist[min_f,:] = np.inf
        
    return np.ndarray.astype(matchings,int)


def match_hungarian(first,second,iou_cutoff = 0.5):
    """
    performs  optimal (in terms of sum distance) matching of points 
    in first to second using the Hungarian algorithm
    inputs - N x 2 arrays of object x and y coordinates from different frames
    output - M x 1 array where index i corresponds to the second frame object 
    matched to the first frame object i
    """
    # find distances between first and second
    dist = np.zeros([len(first),len(second)])
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            dist[i,j] = np.sqrt((first[i,0]-second[j,0])**2 + (first[i,1]-second[j,1])**2)
            
    a, b = linear_sum_assignment(dist)
    
    # convert into expected form
    matchings = np.zeros(len(first))-1
    for idx in range(0,len(a)):
        matchings[a[idx]] = b[idx]
    matchings = np.ndarray.astype(matchings,int)
    
    # calculate intersection over union  (IOU) for all matches
    for i,j in enumerate(matchings):
        x1_left = first[i][0] -first[i][2]*first[i][3]/2
        x2_left = second[j][0] -second[j][2]*second[j][3]/2
        x1_right= first[i][0] + first[i][2]*first[i][3]/2
        x2_right = second[j][0] +second[j][2]*second[j][3]/2
        x_intersection = min(x1_right,x2_right) - max(x1_left,x2_left) 
        
        y1_left = first[i][1] -first[i][2]/2.0
        y2_left = second[j][1] -second[j][2]/2.0
        y1_right= first[i][1] + first[i][2]/2.0
        y2_right = second[j][1] +second[j][2]/2.0
        y_intersection = min(y1_right,y2_right) - max(y1_left,y2_left)
        
        a1 = first[i,3]*first[i,2]**2 
        a2 = second[j,3]*second[j,2]**2 
        intersection = x_intersection*y_intersection
         
        iou = intersection / (a1+a2-intersection) 
        
        # supress matchings with iou below cutoff
        if iou < iou_cutoff:
            matchings[i] = -1
            
    return matchings


def match_all(coords_list,match_fn = match_greedy):
    """
    performs matching using the match_fn strategy for all pairs of consecutive
    coordinate sets in coords_list
    coords_list- list of M x (x,y) pairs
    output - list of matchings between frames
    """
    out_list = []
    for i in range(0,len(coords_list)-1):
        
        first = coords_list[i]
        second = coords_list[i+1]
        out_list.append(match_fn(first,second))
    return out_list


def track_naive(coords,style = "center",snap_threshold = 30, frames_lost_lim = 20):
    """ 
    Tracks objects using the naive assumption that unobserved or detached objects
    do not change position. 
    pt_location = "center" or "bottom_center" and specifies where the object 
    point should be placed
    returns point_array - a t x 2N array where t is the number of frames and N
    is the total number of unique objects detected in the video
    """
    matchings = match_all(coords)   
    
    # track
    active_objs = []
    inactive_objs = []
    # initialize with all objects found in first frame
    for i,row in enumerate(coords[0]):
        obj = {
                'current': (row[0],row[1]),
                'all': [], # keeps track of all positions of the object
                'obj_id': i, # position in coords list
                'fsld': 0,   # frames since last detected
                'first_frame': 0 # frame in which object is first detected
                }
        obj['all'].append(obj['current']) 
        active_objs.append(obj)
    
    # for one set of matchings between frames - this loop will run (# frames -1) times
    # f is the frame number
    # frame_set is the 1 x objects numpy array of matchings from frame f to frame f+1
    for f, frame_set in enumerate(matchings):
        move_to_inactive = []
        matched_in_next = [] # keeps track of which objects from next frame are already dealt with
        
        # first, deal with all existing objects (each loop deals with one object)
        # o is the object index
        # obj is the object 
        for o, obj in enumerate(active_objs):
            matched = False
            obj_id_next = int(frame_set[obj['obj_id']]) # get the index of object in next frame or -1 if not matched in next frame
            # first deals with problem of reverse indexing, second verifies there is a match
            if obj['obj_id'] != -1 and obj_id_next != -1: # object has a match
                obj['obj_id'] = obj_id_next
                obj['current'] = (coords[f+1][obj_id_next,0],coords[f+1][obj_id_next,1])
                obj['all'].append(obj['current'])
                obj['fsld'] = 0
                
                matched_in_next.append(obj_id_next)
                matched = True
                
            else: # object does not have a certain match in next frame
                # search for match among detatched objects
                for j, row in enumerate(coords[f+1]):
                    if j not in matched_in_next: # not already matched to object in previous frame
                        # calculate distance to current obj
                        distance = np.sqrt((obj['current'][0]-row[0])**2 + (obj['current'][1]-row[1])**2)
                        if distance < snap_threshold: # close enough to call a match
                            obj['obj_id'] = j
                            obj['current'] = (row[0],row[1])
                            obj['all'].append(obj['current'])
                            obj['fsld'] = 0
                            
                            matched_in_next.append(j)
                            matched = True
                            break
                # if no match at all
                if not matched:
                    obj['obj_id'] = -1
                    obj['all'].append(obj['current']) # don't update location at all
                    obj['fsld'] += 1
                
                    if obj['fsld'] > frames_lost_lim:
                        move_to_inactive.append(o)
            
        # now, deal with objects found only in second frame  - each row is one object  
        for k, row in enumerate(coords[f+1]):
            # if row was matched in previous frame, the object has already been dealt with
            if k not in matched_in_next:
                # create a new object
                new_obj = {
                        'current': (row[0],row[1]),
                        'all': [], # keeps track of all positions of the object
                        'obj_id': k, # position in coords list
                        'fsld': 0,   # frames since last detected
                        'first_frame': f # frame in which object is first detected
                        }
                new_obj['all'].append(new_obj['current']) 
                active_objs.append(new_obj)
                
        # lastly, move all objects in move_to_inactive
        move_to_inactive.sort()
        move_to_inactive.reverse()
        for idx in move_to_inactive:
            inactive_objs.append(active_objs[idx])
            del active_objs[idx]
            
    objs = active_objs + inactive_objs

    # create an array where each row represents a frame and each two columns represent an object
    points_array = np.zeros([len(coords),len(objs)*2])-1
    for j in range(0,len(objs)):
        obj = objs[j]
        first_frame = int(obj['first_frame'])
        for i in range(0,len(obj['all'])):
            points_array[i+first_frame,j*2] = obj['all'][i][0]
            points_array[i+first_frame,(j*2)+1] = obj['all'][i][1]\
            
    return points_array, objs

class KF_Object():
    """
    A wrapper class that stores a Kalman filter for tracking the object as well
    as some other parameters, variables and all object positions
    """
    def __init__(self, xysr,frame_num,mod_err,meas_err,state_err):
        # use mod_err,meas_err, and state_err to tune filter
        
        self.first_frame = frame_num # first frame in which object is detected
        self.fsld = 0 # frames since last detected
        self.all = [] # all positions of object across frames
        self.tags = []
        self.cls = -1 # object class
        t = 1/30.0
        
        # intialize state (generally x but called state to avoid confusion here)
        state = np.zeros([10,1])
        state[0,0] = xysr[0]
        state[1,0] = xysr[1]
        state[2,0] = xysr[2]
        state[3,0] = xysr[3]

        
        F = np.identity(10) # state transition matrix
        for i in range(0,6):
            F[i,i+4] = t
            
        H = np.zeros([4,10]) # initialize measurement transition matrix
        H[[0,1,2,3],[0,1,2,3]] = 1
        
        second_order = False
        if second_order == True:
            # initialize Kalman Filter to track object
            self.kf = KalmanFilter(dim_x = 10, dim_z = 4)
            self.kf.x = state # state
            self.kf.P *= state_err # state error covariance matrix
            self.kf.Q = np.identity(10)*mod_err # model error covariance matrix
            self.kf.R = np.identity(4)* meas_err # measurement error covariance matrix
            self.kf.F = F
            self.kf.H = H
            


        else:
            # initialize Kalman Filter to track object
            self.kf = KalmanFilter(dim_x = 6, dim_z = 4)
            self.kf.x = state[:6,:] # state
            self.kf.P *= state_err # state error covariance matrix
            self.kf.Q = np.identity(6)*mod_err # model error covariance matrix
            self.kf.R = np.identity(4)* meas_err # measurement error covariance matrix
            self.kf.F = F[:6,:6]
            self.kf.H = H[:,:6] 
            
        # scale errors in r and s so they are comparable to x and y
        self.kf.Q[2,2] = self.kf.Q[2,2] / 10
        self.kf.R[2,2] = self.kf.Q[2,2] / 10
        self.kf.Q[3,3] = self.kf.Q[2,2] / 1000
        self.kf.Q[2,2] = self.kf.Q[2,2] / 1000
        
    def predict(self):
        self.kf.predict()
    
    def update(self,measurement):
        self.kf.update(measurement)
    
    def get_x(self):
        """
        returns current state, so will return a priori state estimate if 
        called after predict, or a posteriori estimate if called after update
        """
        return self.kf.x
    
    def get_coords(self):
        """
        returns 1d numpy array of x,y,s,r
        """
        return self.kf.x[[0,1,2,3],0]
    
    def get_xysr_cov(self):
        covs = np.array([self.kf.P[i,i] for i in [0,1,2,3]])
        return self.kf.x[[0,1,2,3],0], covs

    
def track_SORT(coords_list,mod_err=1,meas_err=1,state_err=100,fsld_max = 60):    
    """
    Uses the SORT algorithm for object tracking. 
    detections - A list of D x 4 numpy arrays with x centroid, y centroid, scale, ratio
    for each object in a frame
    
    objs - returned
    """

    active_objs = []
    inactive_objs = []
    
    # initialize with all objects found in first frame
    for i,row in enumerate(coords_list[0]):
        obj = KF_Object(row,0,mod_err,meas_err,state_err)
        obj.all.append(obj.get_coords())
        obj.cls = row[5] # class
        active_objs.append(obj)

    # loop through all frames
    for frame_num in range(1,len(coords_list)):
        
        # predict new locations of all objects
        # look at next set of detected objects
        # match
        # for all matches, update Kalman filter
        # for all detached objects, update fsld and delete if too high
        # for all unmatched objects, intialize new object
        
        # 1. predict new locations of all objects x_k | x_k-1
        for obj in active_objs:
            obj.predict()
            
        # 2. look at next set of detected objects - all objects are included in this even if detached
        # convert into numpy array - where row i corresponds to object i in active_objs
        locations = np.zeros([len(active_objs),4])
        for i,obj in enumerate(active_objs):
            locations[i,:] = obj.get_coords()
        
        # 3. match - these arrays are both N x 4 (or N x 6) but last two columns will be ignored 
        # remove matches with IOU below threshold (i.e. too far apart)
        second = coords_list[frame_num][:4]
        matches = match_hungarian(locations,second)        
        #matches = match_greedy(locations,second)
        
        # traverse object list
        move_to_inactive = []
        for i in range(0,len(active_objs)):
            obj = active_objs[i]
            
            # update fsld and delete if too high
            if matches[i] == -1:
                obj.fsld += 1
                obj.all.append(obj.get_coords())
                obj.tags.append(0) # indicates object not detected in this frame
                if obj.fsld > fsld_max or (obj.fsld > 0 and len(obj.all)< 3):
                    move_to_inactive.append(i)
                    
                
            else: # object was matched        
                # update Kalman filter
                measure_coords = second[matches[i]][:4]
                obj.update(measure_coords)
                obj.fsld = 0
                obj.all.append(obj.get_coords())
                obj.tags.append(1) # indicates object detected in this frame

        # for all unmatched objects, intialize new object
        for j in range(0,len(second)):
            if j not in matches:
                new_obj = KF_Object(second[j][:4],frame_num,mod_err,meas_err,state_err)
                new_obj.all.append(new_obj.get_coords())
                new_obj.tags.append(1) # indicates object detected in this frame
                new_obj.cls = second[j,5] # class
                active_objs.append(new_obj)

        
        # move all necessary objects to inactive list
        move_to_inactive.sort()
        move_to_inactive.reverse()
        for idx in move_to_inactive:
            inactive_objs.append(active_objs[idx])
            del active_objs[idx]
            
            
    objs = active_objs + inactive_objs
    
    # create final point array
    points_array = np.zeros([len(coords_list),len(objs)*2])-1
    for j in range(0,len(objs)):
        obj = objs[j]
        if len(obj.all) > 3: # only keep object that were found in first 3 frames
            first_frame = int(obj.first_frame)
            for i in range(0,len(obj.all)):
                points_array[i+first_frame,j*2] = obj.all[i][0]
                points_array[i+first_frame,(j*2)+1] = obj.all[i][1]
    
    return objs, points_array
    

if __name__ == "__main__":
    # Kalman filter validation code
    detections = np.load("temp_detections.npy",allow_pickle= True)
    flattened = condense_detections(detections,style = "SORT")
    objs,point_array = track_SORT(flattened)

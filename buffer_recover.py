# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:27:36 2018

@author: hzhou
"""

import numpy as np  # numpy
from sklearn.utils.linear_assignment_ import linear_assignment
import math


def dis(bb_test,bb_gt):
    """
    computes distance between two centres
    """
    xx1 = bb_test[1]+bb_test[3]/2
    yy1 = bb_test[2]+bb_test[4]/2
    xx2 = bb_gt[1]+bb_gt[3]/2
    yy2 = bb_gt[2]+bb_gt[4]/2
    diffx = xx2 - xx1
    diffy = yy2 - yy1
    diff_dis = math.sqrt(diffx*diffx +diffy*diffy)
    return(diff_dis)
    
def cls(bb_test,bb_gt):
    """
    computes the class difference between two predictions(Apearance information)
    """
    cls1=bb_test[0]
    cls2=bb_gt[0]
    diff_cls=cls2-cls1
    return(diff_cls)
    
def conf(bb_test,bb_gt):
    """
    computes the prediction confidence difference between two bounding boxes(Integrating with the apearance information)
    """
    conf1=bb_test[5]
    conf2=bb_gt[5]
    diff_conf=conf2-conf1
    return(diff_conf)

def obtain_cur_det1(raw_det,frame_counter,start_index=1):
    """
    Obtain the current detections at a specific frame and extract index number from the starting index to the very end column
    """
    raw_now = raw_det[raw_det[:,0]==frame_counter,start_index:7]
    raw_nex = raw_det[raw_det[:,0]==frame_counter+1,start_index:7]
    return raw_now,raw_nex

def calc_matrix(raw_now,raw_nex):
    """
    Reset and calculate the constaints matrix by using the previously defined functions
    """
    dis_matrix = np.zeros((len(raw_now),len( raw_nex)),dtype=np.float32)
    class_matrix = np.zeros((len(raw_now),len( raw_nex)),dtype=np.float32)
    conf_matrix = np.zeros((len(raw_now),len( raw_nex)),dtype=np.float32)    
    for d,now in enumerate(raw_now):
        for t,nex in enumerate(raw_nex):
            dis_matrix[d,t] = dis(now,nex)
            class_matrix[d,t] = cls(now,nex)
            conf_matrix[d,t] = conf(now,nex)
    return dis_matrix,class_matrix,conf_matrix

def not_in_constraints(m,dis_matrix,class_matrix,conf_matrix,dis_threshold,conf_threshold):
    """
    Generates a flag which indicates whether the Hungarian algorithm matched indices fit our constraints or not.
    """
    wrong_match_flag=False
    if(dis_matrix[m[0],m[1]]>dis_threshold or class_matrix[m[0],m[1]] !=0 or conf_matrix[m[0],m[1]]>conf_threshold):
        wrong_match_flag=True
    return wrong_match_flag

def list_unmatched_now(raw_now,matched_indices):
    """
    List the indexes directing to all unmatched rows
    """
    unmatched_now = []
    for d,now in enumerate(raw_now):
        if(d not in matched_indices[:,0]):
            unmatched_now.append(d)
    return unmatched_now

def list_unmatched_nex(raw_nex,matched_indices):
    unmatched_nex = []
    for t,nex in enumerate(raw_nex):
        if(t not in matched_indices[:,1]):
            unmatched_nex.append(t)
    return unmatched_nex

def update_unmatch(unmatched_now,raw_now):
    """
    Generate an internal array extracting from the external detection array in the frame by using the index from the unmatched list
    """
    raw_unmatch=np.empty((0,raw_now.shape[1]+1),np.float32)
    if len(unmatched_now)>0:    
        for i,index in enumerate(unmatched_now):
            raw_unmatch_tmp=raw_now[index,:]
            raw_unmatch_tmp=np.append(raw_unmatch_tmp,index) # append a new column which contains the original of index to the new detection array
            raw_unmatch=np.vstack([raw_unmatch,raw_unmatch_tmp])
            
    return raw_unmatch

def obtain_cur_det0(raw_det,match_det,frame_counter):
    """
    Obtain the current detections at a specific frame and extract index number from the starting index to the very end column
    """
    raw_now = raw_det[raw_det[:,0]==frame_counter,0:7]
    raw_nex = raw_det[raw_det[:,0]==frame_counter+1,0:7]
    #print(raw_now)
    match_now=match_det[match_det[:,0]==frame_counter,0:4]
    match_nex=match_det[match_det[:,0]==frame_counter+1,0:4]
    match_prev=match_det[match_det[:,0]==frame_counter-1,0:4]
    #unmatch=unmatch_det[unmatch_det[:,0]==frame_counter,1:2]
    return raw_now,raw_nex,match_now,match_nex,match_prev

def write_output(raw_now,det_counter,obj_id,out_file):
    out_file.write("%d,%d,%d,%d,%d,%d,%s,%d\n"%(raw_now[det_counter,0],raw_now[det_counter,1],raw_now[det_counter,2],raw_now[det_counter,3],raw_now[det_counter,4],raw_now[det_counter,5],raw_now[det_counter,6],obj_id))

def next_frame_match(raw_det,match_det,frame_counter,match_thres,det_counter):
    """
    In this section, a treak is used which indicates whether there is a mismatch by setting a threshold. In this example, index 10000 is used to show a mismatch
    """
    next_fm_flag=False
    _,_,match_now,_,_=obtain_cur_det0(raw_det,match_det,frame_counter)
    if match_now[det_counter,3]!=match_thres:
        next_fm_flag=True
    return next_fm_flag,match_now

def no_prev_match(det_counter,frame_counter,frame_story,raw_det,match_det):
    """
    Check the previous frame stored in the memory and generate the flag if there is no matched object in previously stored frames.
    """
    no_pm_flag=True
    _,_,match_now,_,_=obtain_cur_det0(raw_det,match_det,frame_counter)
    if frame_counter-frame_story<int(raw_det[0,0]):
        frame_story=frame_counter-int(raw_det[0,0])
    for frame in range (frame_counter-frame_story,frame_counter):
        _,_,match_prev,_,_=obtain_cur_det0(raw_det,match_det,frame)
        for match_counter in range(len(match_prev)):
            if match_prev[match_counter,3]==match_now[det_counter,1] and match_prev[match_counter,2]==frame_counter:
                no_pm_flag=False
    return no_pm_flag

detections="resultextra.txt"
#define the threshold
dis_threshold=50
conf_threshold=1
frame_store=10

if type(detections) is str:
    raw_det = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
else:
    assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
    raw_det = detections.astype(np.float32)    

obj_id=1
i=1       
frame_matches=[]

with open("out_matches.txt", "w") as match_file:
    for frame_counter in range(int(raw_det[:,0].max())):
        raw_now,raw_nex=obtain_cur_det1(raw_det,frame_counter)
        if len(raw_now)>1:
            dis_matrix,class_matrix,conf_matrix=calc_matrix(raw_now,raw_nex)
            matched_indices = linear_assignment(dis_matrix)
            
            unmatched_now = list_unmatched_now(raw_now,matched_indices)
            unmatched_nex = list_unmatched_nex(raw_nex,matched_indices)
                #match_file.write("%d,%d,%d,1000\n"%(frame_counter,d,frame_counter+1))
            #filter out matched with low IOU
            matches = []
            for m in matched_indices:
                if not_in_constraints(m,dis_matrix,class_matrix,conf_matrix,dis_threshold,conf_threshold):
                    #match_file.write("%d,%d,%d,1000\n"%(frame_counter,m[0],frame_counter+1))
                    unmatched_now.append(m[0])
                    unmatched_nex.append(m[1])
                else:
                    match_file.write("%d,%d,%d,%d\n"%(frame_counter,m[0],frame_counter+1,m[1]))
                    matches.append(m.reshape(1,2))
                    
            raw_unmatch=update_unmatch(unmatched_now,raw_now)
        
            if len(raw_unmatch)>0:
                if frame_counter+frame_store<int(raw_det[:,0].max()):
                    for next_frame_counter in range(frame_counter+1,frame_counter+frame_store):
                        
                        _,raw_nex=obtain_cur_det1(raw_det,next_frame_counter)
                        dis_matrix,class_matrix,conf_matrix=calc_matrix(raw_unmatch,raw_nex)
                        matched_indices = linear_assignment(dis_matrix)
                        unmatched_now = list_unmatched_now(raw_unmatch,matched_indices)
                        for m in matched_indices:
                            if not_in_constraints(m,dis_matrix,class_matrix,conf_matrix,dis_threshold,conf_threshold):
                                unmatched_now.append(m[0])
                            else:
                                match_file.write("%d,%d,%d,%d\n"%(frame_counter,raw_unmatch[m[0],6],next_frame_counter+1,m[1]))
                                matches.append(m.reshape(1,2))
                        raw_unmatch_old=raw_unmatch
                        raw_unmatch=update_unmatch(unmatched_now,raw_unmatch)
                    
                    if len(unmatched_now)>0:
                        for i,index in enumerate(unmatched_now):
                            match_file.write("%d,%d,%d,-1\n"%(frame_counter,raw_unmatch_old[index,6],frame_counter+1))
                else:
                    if len(unmatched_now)>0:
                        for i,index in enumerate(unmatched_now):
                            match_file.write("%d,%d,%d,-1\n"%(frame_counter,raw_unmatch[i,6],frame_counter+1))
                            
match_det = np.genfromtxt("out_matches.txt", delimiter=',', dtype=int,names=["a", "b", "c","d"])
match_det=np.sort(match_det, order=['a', 'b']) 

np.savetxt("sortmatch.txt", match_det, delimiter=',')
match_det = np.genfromtxt("sortmatch.txt", delimiter=',', dtype=np.float32)

loop=True
while loop:
    loop=False
    for frame_counter in range(1,int(match_det[:,0].max())):
        match_now=match_det[match_det[:,0]==frame_counter,0:4]
        for i in range(len(match_now)):
            frame_gap=match_now[i,2]-match_now[i,0]
            if frame_gap>1:
                for repeat_counter in range(int(frame_gap)-1):    
                    
                    start_frame=int(match_now[i,0])
                    end_frame=int(match_now[i,2])
                    for find_counter in range(start_frame+1,end_frame):
                        match_new=match_det[match_det[:,0]==find_counter,0:4]
                        for j in range(len(match_new)):
                            if match_new[j,2]==match_now[i,2] and match_new[j,3]==match_now[i,3]:
                                match_now[i,2]=match_new[j,0]
                                match_now[i,3]=match_new[j,1]
                                loop=True
                                break
                    if match_now[i,2]-match_now[i,0]==1:
                        break
            else:
                for j in range(len(match_now)):
                    if match_now[j,2]==match_now[i,2] and match_now[j,3]==match_now[i,3] and match_now[j,1]!=match_now[i,1]:
                                match_now[i,2]=frame_counter+1
                                match_now[i,3]=-1
#                                match_now[j,2]=match_now[i,0]
#                                match_now[j,3]=match_now[i,1]
                                
        match_det[match_det[:,0]==frame_counter,0:4]=match_now
    
#for frame_counter in range()


np.savetxt("sortmatch.txt", match_det, delimiter=',')


raw_det = np.genfromtxt("resultextra.txt", delimiter=',', dtype=np.float32)
match_det = np.genfromtxt("sortmatch.txt", delimiter=',', dtype=np.float32)
obj_id = 1
match_thres = -1

with open("out_det_matched.txt", "w") as out_file:
    for frame_counter in range(1,int(raw_det[:,0].max())-10):
        raw_now,_,_,_,_=obtain_cur_det0(raw_det,match_det,frame_counter)
        for det_counter in range(len(raw_now)):#loop for each index in the detections in the current frame
            if frame_counter==int(raw_det[0,0]): #for the starting frame
                write_output(raw_now,det_counter,obj_id,out_file)
                next_frame_flag,match_now = next_frame_match(raw_det,match_det,frame_counter,match_thres,det_counter)
                match_counter_tmp=det_counter
                raw_new=raw_now
                while next_frame_flag:
                    frame_counter_tmp=match_now[match_counter_tmp,2]
                    match_counter_tmp=int(match_now[match_counter_tmp,3])
                    raw_new,_,_,_,_=obtain_cur_det0(raw_det,match_det,frame_counter_tmp)
                    write_output(raw_new,match_counter_tmp,obj_id,out_file)
                    next_frame_flag,match_now = next_frame_match(raw_det,match_det,frame_counter_tmp,match_thres,match_counter_tmp)
                obj_id+=1
            else:    
                if frame_counter>int(raw_det[0,0]):
                    if no_prev_match(det_counter,frame_counter,frame_store,raw_det,match_det):#if there is no previous match for the current detection counter in the current frame, create new object
                        obj_id+=1
                        write_output(raw_now,det_counter,obj_id,out_file)
                        next_frame_flag,match_now = next_frame_match(raw_det,match_det,frame_counter,match_thres,det_counter)
                        match_counter_tmp=det_counter
                        raw_new=raw_now
                        while next_frame_flag:
                            frame_counter_tmp=match_now[match_counter_tmp,2]
                            match_counter_tmp=int(match_now[match_counter_tmp,3])
                            raw_new,_,_,_,_=obtain_cur_det0(raw_det,match_det,frame_counter_tmp)
                            write_output(raw_new,match_counter_tmp,obj_id,out_file)
                            if frame_counter_tmp < int(raw_det[:,0].max())-1: #this constraint solves the last frame problem
                                next_frame_flag,match_now = next_frame_match(raw_det,match_det,frame_counter_tmp,match_thres,match_counter_tmp)
                            else:
                                next_frame_flag=False

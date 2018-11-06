# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:00:44 2018

@author: hzhou
"""



from filterpy.kalman import KalmanFilter
import numpy as np
import cv2

def convert_x_to_bbox(x,obj_id,frame=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(frame==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([frame,x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,obj_id]).reshape((1,6))


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]
  h = bbox[3]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def tracker(cap,detections):
    frame_counter=0
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    else:
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)            
    while True:
        ret, img = cap.read()
    # if video stopped playing, quit
        if ret == False:
            break
        frame_counter=frame_counter+1
        for i in range(1, len(raw)):    
           # print(raw[i,0], " ",frame_counter)
            if raw[i,0,0] == frame_counter:

                pred_xmin = raw[i,0,1]
                pred_ymin = raw[i,0,2]
                pred_xmax = raw[i,0,3]
                pred_ymax = raw[i,0,4]
                
                if np.isnan(pred_xmin)!=True and np.isnan(pred_ymin)!=True and np.isnan(pred_xmax)!=True and np.isnan(pred_ymax)!=True:
                    cv2.rectangle(img,
                                  (pred_xmin, pred_ymin),
                                  (pred_xmax, pred_ymax),
                                  (0, 255, 50), 2)
                    text_x = int(pred_xmin) - 10
                    text_y = int(pred_ymin) - 10
                    obj_id=raw[i,0,5]
                    cv2.putText(img, str(obj_id)+' ', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    
        cv2.imshow('Image', img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        
def KalmanTracker(bbox):
    kf = KalmanFilter(dim_x=10, dim_z=4)
    kf.F = np.array([[1,0,0,0,1,0,0,0.5,0,0],[0,1,0,0,0,1,0,0,0.5,0],[0,0,1,0,0,0,1,0,0,0.5],[0,0,0,1,0,0,0,0,0,0],  [0,0,0,0,1,0,0,1,0,0],[0,0,0,0,0,1,0,0,1,0],[0,0,0,0,0,0,1,0,0,1],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])
    kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0]])
    kf.R[2:,2:] *= 10.
    kf.P[4:,4:] *= 10. 
    kf.P[7:,7:] *= 1000. 
#    kf.P *= 10.
    kf.Q[-1,-1] *= 0.01
    kf.Q[4:,4:] *= 0.01
    kf.Q[7:,7:] *= 0.01
    kf.x[:4] = convert_bbox_to_z(bbox)

    return kf

detections="out_det_matched.txt"
video = cv2.VideoCapture('workhall.mp4')
det = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
xf=[]
for obj_id in range(1,int(det[:,7].max())+1):
    obj_det=det[det[:,7]==obj_id,0:6]
#    obj_det[:,4:6] += obj_det[:,2:4] 
    if len(obj_det)>0:
        kf=KalmanTracker(obj_det[0,2:6])
        
        for frame_counter in range(int(obj_det[0,0]),int(obj_det[:,0].max())+1):
            det_now=obj_det[obj_det[:,0]==frame_counter,0:6]
            if len(det_now)>0:
                frame=det_now[0,0]
                det_now=det_now[0,2:6]
                kf.predict()
                kf.update(convert_bbox_to_z(det_now))
                xf.append(convert_x_to_bbox(kf.x,obj_id,frame))
            else:
                frame=frame_counter
                kf.predict()
                xf.append(convert_x_to_bbox(kf.x,obj_id,frame))
                
xf=np.array(xf)
#print(xf)
tracker(video,xf)
video.release()
cv2.destroyAllWindows
    
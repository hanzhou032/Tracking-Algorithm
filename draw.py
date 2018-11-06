# libraries that will be needed
import numpy as np  # numpy
import cv2          # opencv
from sort import *  # sort

# function to trak people
def tracker(cap,detections):
    frame_counter=0
    
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
        print(np.shape(raw))
    else:
    # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)            
    while True:
        ret, img = cap.read()
    # if video stopped playing, quit
        if ret == False:
            break
        
        frame_counter=frame_counter+1
        #print(len(raw[:,0]))
        for i in range(1, len(raw[:,0])):    
           # print(raw[i,0], " ",frame_counter)
            if raw[i,0] == frame_counter:
                 
                conf = raw[i,6]
            
                label = raw[i,7]
    
                pred_xmin = raw[i,2]
                pred_ymin = raw[i,3]
                pred_xmax = raw[i,2]+raw[i,4]
                pred_ymax = raw[i,3]+raw[i,5]
        
                cv2.rectangle(img,
                              (pred_xmin, pred_ymin),
                              (pred_xmax, pred_ymax),
                              (0, 255, 50), 2)
                text_x = int(pred_xmin) - 10
                text_y = int(pred_ymin) - 10
                
                #trackers = mot_tracker.update(dets)
                #print(pred_xmin)
                
                # show image#
                #if detections[i]['new_frame'] == True:
                   # cv2.putText(img, "New: " +  str(label)+' '+str(conf), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                
                cv2.putText(img, ' '+str(conf), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                
        cv2.imshow('Image', img)
        
            #else:
                #print("wrong")
        
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
# run video        
#detections="iou.txt"
detections="resultextra.txt"
#detections="out_det_matched.txt"
cap = cv2.VideoCapture('workhall.mp4')
tracker(cap,detections)
# release frame and destroy windows
cap.release()
cv2.destroyAllWindows
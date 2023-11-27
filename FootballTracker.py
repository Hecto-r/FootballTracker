######## Webcam Object Detection Using Tensorflow #######################
#
# Author: Hector Cabrera
# Date: 11/25/2023
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.

import cv2
import time
import random
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from ocsort import ocsort
from Utils import utils
from Utils.VideoStream import VideoStream

model='Models/edgetpu.tflite'
mIsTrackEnabled = True

base_option=core.BaseOptions(file_name=model,use_coral=True, num_threads=4)
detection_options=processor.DetectionOptions(max_results=8, score_threshold=.5)
options=vision.ObjectDetectorOptions(base_options=base_option, detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#Initialize OCSort
tracker = ocsort.OCSort(det_thresh=0.30, max_age=10, min_hits=5)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Initialize video stream
videostream = VideoStream(resolution=(640,480),framerate=30).start()
time.sleep(1)

while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame = cv2.flip(frame,-1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640,480))
    
    imTensor=vision.TensorImage.create_from_array(frame_resized)
    results=detector.detect(imTensor)
    #image=utils.visualize(frame, results) # Default BBox
    
    for detection in range(len(results.detections)):
        detections = []
        for result in results.detections:
            x1 = int(result.bounding_box.origin_x)
            x2 = int(result.bounding_box.origin_x+result.bounding_box.width)
            y1 = int(result.bounding_box.origin_y)
            y2 = int(result.bounding_box.origin_y+result.bounding_box.height)
            class_id = int(result.categories[0].index)
            score = result.categories[0].score
            if score > .5:
                detections.append([x1, y1, x2, y2, score, class_id])
                
        if mIsTrackEnabled and detections:
            detections = np.array(detections)
            tracker.update(detections, frame)
            for track in tracker.trackers:
                track_id = track.id
                hits = track.hits
                color = colors[track_id % len(colors)]
                x1,y1,x2,y2 = np.round(track.get_state()).astype(int).squeeze()

                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                cv2.putText(frame, 
                  f"{track_id}-{hits}", 
                  (x1+10,y1-5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  0.5,
                  color, 
                  1,
                  cv2.LINE_AA)
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Camera',frame)
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
videostream.stop()
######## Webcam Object Detection Using Tensorflow #######################
#
# Author: Hector Cabrera
# Date: 11/28/2023
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.

import cv2
import time
import random
import numpy as np
import sys
import os
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from ocsort import ocsort
from Utils import utils
from Utils.VideoStream import VideoStream
from Utils.PanTiltController import PanTiltController
from config import *

cam_pan = INITIAL_PAN
cam_tilt = INITIAL_TILT

base_option=core.BaseOptions(file_name=MODEL_FILE, use_coral=True, num_threads=4)
detection_options=processor.DetectionOptions(max_results=MAX_RESULTS, score_threshold=SCORE_THRESHOLD)
options=vision.ObjectDetectorOptions(base_options=base_option, detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#Initialize OCSort
tracker = ocsort.OCSort(det_thresh=0.30, max_age=30, min_hits=2)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Initialize video stream
videostream = VideoStream(resolution=(DISPLAY_W,DISPLAY_H),framerate=30).start()
mPanTiltController = PanTiltController().start()
time.sleep(1)
tic=0

while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame1 = videostream.read()
    frame = frame1.copy()
    frame = cv2.flip(frame,-1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (DISPLAY_W,DISPLAY_H))
    
    imTensor=vision.TensorImage.create_from_array(frame_resized)
    results=detector.detect(imTensor)
    #image=utils.visualize(frame, results) # Default BBox
    
    toc = time.perf_counter()
    totalTime = toc-tic
    if totalTime > 3:
        mPanTiltController.scan(inBool=True)
    
    for detection in range(len(results.detections)):
        detections = []
        for result in results.detections:
            x1 = int(result.bounding_box.origin_x)
            x2 = int(result.bounding_box.origin_x+result.bounding_box.width)
            y1 = int(result.bounding_box.origin_y)
            y2 = int(result.bounding_box.origin_y+result.bounding_box.height)
            class_id = int(result.categories[0].index)
            category_name = result.categories[0].category_name
            score = result.categories[0].score
            if score > .5 and class_id == 0:
                cv2.rectangle(frame, (x1,y1),(x2,y2), COLOR, PAINT_WEIGHT)
                detections.append([x1, y1, x2, y2, score, class_id])
                
                label = '%s: %d%%' % (category_name, int(score*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, PAINT_WEIGHT) # Get font size
                label_ymin = max(y1, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (x1, label_ymin-labelSize[1]-10), (x1+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (x1, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 0), PAINT_WEIGHT) # Draw label text
                
                mPanTiltController.scan(inBool=False)
                mPanTiltController.trackObject(x1, y1, x2, y2)
                tic = time.perf_counter()
                
        if IS_TRACK_ENABLED and detections:
            detections = np.array(detections)
            tracker.update(detections, frame)
            for track in tracker.trackers:
                track_id = track.id
                hits = track.hits
                color = colors[track_id % len(colors)]
                x1,y1,x2,y2 = np.round(track.get_state()).astype(int).squeeze()

                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                cv2.putText(frame,f"{track_id}-{hits}", (x1+10,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(5,20), cv2.FONT_HERSHEY_SIMPLEX,.7, (255,255,0), PAINT_WEIGHT, cv2.LINE_AA)
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
mPanTiltController.scan(inBool=False)
sys.exit("Exiting the code!")
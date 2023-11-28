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
import sys
import os
import pantilthat
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from ocsort import ocsort
from Utils import utils
from Utils.VideoStream import VideoStream

# Find the full path of this python script
SCRIPT_PATH = os.path.abspath(__file__)
# get the path location only (excluding script name)
SCRIPT_DIR = SCRIPT_PATH[0:SCRIPT_PATH.rfind("/")+1]
baseFileName = SCRIPT_PATH[SCRIPT_PATH.rfind("/")+1:SCRIPT_PATH.rfind(".")]
# Read Configuration variables from config.py file
configFilePath = SCRIPT_DIR + "config.py"
if not os.path.exists(configFilePath):
    print("ERROR - Missing config.py file - Could not find Configuration file %s"
          % (configFilePath))
    import urllib2
    config_url = "https://raw.github.com/pageauc/face-track-demo/master/config.py"
    print("   Attempting to Download config.py file from %s" % config_url)
    try:
        wgetfile = urllib2.urlopen(config_url)
    except:
        print("ERROR - Download of config.py Failed")
        print("        Try Rerunning the face-track-install.sh Again.")
        print("        or")
        print("        Perform GitHub curl install per Readme.md")
        print("        and Try Again")
        print("Exiting %s" % PROG_NAME)
        quit()
    f = open('config.py', 'wb')
    f.write(wgetfile.read())
    f.close()
from config import *

cam_pan = INITIAL_PAN
cam_tilt = INITIAL_TILT
# Turn the camera to the Start position (the data that pan() and tilt() functions expect to see are any numbers between -90 to 90 degrees).
pantilthat.pan(INITIAL_PAN-90)
pantilthat.tilt(INITIAL_TILT-90)

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
time.sleep(1)
tic=0
while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame = cv2.flip(frame,-1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (DISPLAY_W,DISPLAY_H))
    
    imTensor=vision.TensorImage.create_from_array(frame_resized)
    results=detector.detect(imTensor)
    image=utils.visualize(frame, results) # Default BBox
    
    for detection in range(len(results.detections)):
        detections = []
        for result in results.detections:
            x1 = int(result.bounding_box.origin_x)
            x2 = int(result.bounding_box.origin_x+result.bounding_box.width)
            y1 = int(result.bounding_box.origin_y)
            y2 = int(result.bounding_box.origin_y+result.bounding_box.height)
            class_id = int(result.categories[0].index)
            score = result.categories[0].score
            if score > .5 and class_id == 0:
                detections.append([x1, y1, x2, y2, score, class_id])
                x = x1 + ((x2-x1)/2)
                y = y1 + ((y2-y1)/2)

                # Correct relative to centre of image
                turn_x  = float(x - (DISPLAY_W/2))
                turn_y  = float(y - (DISPLAY_H/2))

                # Convert to percentage offset
                turn_x  /= float(DISPLAY_W/2)
                turn_y  /= float(DISPLAY_H/2)

                # Scale offset to degrees (that 2.5 value below acts like the Proportional factor in PID)
                turn_x   *= 2.5 # VFOV
                turn_y   *= 2.5 # HFOV
                cam_pan  += -turn_x
                cam_tilt += turn_y

                # Clamp Pan/Tilt to 0 to 180 degrees
                cam_pan = max(0,min(180,cam_pan))
                cam_tilt = max(0,min(180,cam_tilt))

                # Update the servos
                pantilthat.pan(int(cam_pan-90))
                pantilthat.tilt(int(cam_tilt-90))
                tic = time.perf_counter()
                
            toc = time.perf_counter()
            totalTime = toc-tic
            if totalTime > 3:
                pantilthat.pan(INITIAL_PAN-90)
                pantilthat.tilt(INITIAL_TILT-90)
                cam_pan = INITIAL_PAN
                cam_tilt = INITIAL_TILT
                
                
        if IS_TRACK_ENABLED and detections:
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
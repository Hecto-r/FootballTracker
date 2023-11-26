import cv2
import time
from threading import Thread
from picamera2 import Picamera2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.picam2=Picamera2()
        self.picam2.preview_configuration.main.size=(resolution[0], resolution[1])
        self.picam2.preview_configuration.main.format='RGB888'
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
            
        # Read first frame from the stream
        self.frame = self.picam2.capture_array()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.picam2.stop()
                return

            # Otherwise, grab the next frame from the stream
            self.frame = self.picam2.capture_array()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

model='Models/edgetpu.tflite'

pos=(20,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(255,0,0)
fps=0

base_option=core.BaseOptions(file_name=model,use_coral=True, num_threads=4)
detection_options=processor.DetectionOptions(max_results=8, score_threshold=.3)
options=vision.ObjectDetectorOptions(base_options=base_option, detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

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
    detections=detector.detect(imTensor)
    image=utils.visualize(frame, detections)
    
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
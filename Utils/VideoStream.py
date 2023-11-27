from threading import Thread
from picamera2 import Picamera2

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
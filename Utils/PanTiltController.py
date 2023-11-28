from threading import Thread
import pantilthat
import sys
import time
sys.path.append('/home/admin/Repos/FootballTracker/')
from config import *

class PanTiltController:
    """Object that controls Pan/Tilt"""
    def __init__(self):
        pantilthat.pan(INITIAL_PAN-90)
        pantilthat.tilt(INITIAL_TILT-90)
        self.pan = INITIAL_PAN
        self.tilt = INITIAL_TILT
        self.scanning = False

    def start(self):
        Thread(target=self.scanningThread,args=()).start()
        return self

    def scanningThread(self):
        while True:
            if self.scanning:
                self.pan = self.pan + pan_move_x
                if self.pan > pan_max_right:
                    self.pan = pan_max_left
                # Update the servos
                pantilthat.pan(int(self.pan-90))
            time.sleep(1)
            

    def scan(self, inBool):
	# Indicate that the camera and thread should be stopped
        self.scanning = inBool
        
    def trackObject(self, x1, y1, x2, y2):
	# Indicate that the camera and thread should be stopped
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
        self.pan  += -turn_x
        self.tilt += turn_y

        # Clamp Pan/Tilt to 0 to 180 degrees
        self.pan = max(0,min(180,self.pan))
        self.tilt = max(0,min(180,self.tilt))

        # Update the servos
        pantilthat.pan(int(self.pan-90))
        pantilthat.tilt(int(self.tilt-90))

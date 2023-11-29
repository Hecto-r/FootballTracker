# Config.py file for face-track.py ver 0.94

# Settings

MODEL_FILE='Models/edgetpu.tflite'
SCORE_THRESHOLD = .5
MAX_RESULTS = 8
IS_TRACK_ENABLED = False
DISPLAY_W = 640
DISPLAY_H = 480
INITIAL_PAN = 90
INITIAL_TILT = 90
COLOR = (10, 255, 0)
FONT_SIZE = .6
PAINT_WEIGHT = 2

pan_max_left = 0
pan_max_right = 180
pan_max_top = 30
pan_max_bottom = 70
pan_move_x = 10  # Amount to pan left/right in search mode
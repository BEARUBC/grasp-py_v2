import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.exposure import match_histograms
import cv2

def colour_limits(c, l_tol, a_tol, b_tol):
    lower_limit = np.array([max(0, c[0] - l_tol), max(1, c[1] - a_tol), max(1, c[2] - b_tol)], np.uint8)
    upper_imit = np.array([min(255, c[0] + l_tol), min(255, c[1] + a_tol), min(255, c[2] + b_tol)], np.uint8)

    return lower_limit, upper_imit

'''
Parameters
----------
webcam : video capture object
    Current webcam
Returns
---------
red :  
'''
def calibrate():
    webcam = cv2.VideoCapture(0)
  
    # Start a while loop 
    while(1): 
        _, frame = webcam.read()
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        height, width, _ = frame.shape

        # locate centre
        cx = int(width / 2)
        cy = int(height / 2)
        cv2.circle(frame, (cx, cy),  5, (255, 0, 0), 3)
        
        cv2.imshow("calibration", frame)
        key = cv2.waitKey(1)
        if key == 114: # pressed r
            red = lab_frame[cy, cx]
            print(red)
        if key == 103: # pressed g
            green = lab_frame[cy, cx]
            print(green)
        if key == 98: # pressed b
            blue = lab_frame[cy, cx]
            print(blue)
        if key == 32: # pressed space
            webcam.release() 
            cv2.destroyAllWindows() 
            break
    return red, green, blue

        

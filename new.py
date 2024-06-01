# Python code for Multiple Color Detection 
  
  
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.exposure import match_histograms
import cv2
import imutils
from util2 import calibrate
from util2 import colour_limits

BORDER_COLOUR = (255, 0, 0)
BORDER_THICKNESS = 3
CENTRE_COLOUR = (0, 255, 0)
CENTRE_RADIUS = 5
CAL_RADIUS = 5
CAL_THICKNESS = 2
CAL_COLOR = (255, 0 , 0)
TEXT_COLOR = (0, 255, 0)
TEXT_SCALE = 0.5
THRESHOLD = 100 # can be adjusted to determine the distance at which we should identify objects

L_TOL = 80
A_TOL = 10
B_TOL = 10

# Default r based on previous methods
r = np.array([int((190 - 20)/2) + 20, int((255 - 150)/2) + 150, int((255 - 150)/2) + 150])

# Capturing video through webcam 
webcam = cv2.VideoCapture(0) 
  
# Start a while loop 
while(1): 
      
    # Reading the video from the 
    # webcam in image frames 
    _, imageFrame = webcam.read() 

    # locate the screen
    height, width, _ = imageFrame.shape
    screen_w = int(width/4)
    screen_h = int(screen_w/4*5.2)
    screen_x = int(width/2-screen_w/2)
    screen_y = int(height/2-screen_h/1.5)

    # locate first button
    b1_w = int(screen_w/4*0.5)
    b1_h = b1_w
    b1_x = int(screen_x - 1.5*b1_w)
    b1_y = int(screen_y + screen_h/5.2*0.55)
    b2_w = b1_w
    b2_h = b1_w
    b2_x = b1_x
    b2_y = b1_y + 2*b1_h

    cv2.rectangle(imageFrame, (screen_x, screen_y), (screen_x + screen_w, screen_y + screen_h), (0, 255, 0), 2)
    cv2.rectangle(imageFrame, (b1_x, b1_y), (b1_x + b1_w, b1_y + b1_h), (0, 255, 0), 2)
    cv2.rectangle(imageFrame, (b2_x, b2_y), (b2_x + b2_w, b2_y + b2_h), (0, 255, 0), 2)

    # calibration point
    cal_x = int(width/2)
    cal_y = int(screen_y + screen_h + 30)
    cv2.circle(imageFrame, (cal_x, cal_y), CAL_RADIUS, CAL_COLOR, CAL_THICKNESS)
    cv2.putText(imageFrame, "Press 'c' to calibrate!", (cal_x, cal_y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, 1)

    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # LAB color space
    blurFrame = cv2.bilateralFilter(imageFrame, 10, 100, 100)
    labFrame = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2LAB)
    

    # calibrate
    if cv2.waitKey(1) == 99:  # press c
        r = labFrame[cal_y, cal_x]
        print(r)
  
    # Set range for red color and define mask
    red_lower, red_upper = colour_limits(r, L_TOL, A_TOL, B_TOL)
    red_mask = cv2.inRange(labFrame, red_lower, red_upper)
  
    # # Set range for green color and  
    # # define mask 
    # green_lower, green_upper = colour_limits(g, h_tol, s_tol, v_tol)
    # green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 
  
    # # Set range for blue color and 
    # # define mask 
    # blue_lower, blue_upper = colour_limits(b, h_tol, s_tol, v_tol) 
    # blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
      
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between imageFrame and mask determines 
    # to detect only that particular color 
    kernel = np.ones((5, 5), "uint8") 
      
    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    # res_red = cv2.bitwise_and(labFrame, labFrame,
    #                           mask = red_mask)
      
    # For green color 
    # green_mask = cv2.dilate(green_mask, kernel) 
    # res_green = cv2.bitwise_and(imageFrame, imageFrame, 
    #                             mask = green_mask) 
      
    # # For blue color 
    # blue_mask = cv2.dilate(blue_mask, kernel) 
    # res_blue = cv2.bitwise_and(imageFrame, imageFrame, 
    #                            mask = blue_mask) 
   
    # Creating contour to track red color 
    contours = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if (cv2.contourArea(contour) > THRESHOLD):
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-5))  # calculate X position and add 1e-5 to avoid division by 0
            cy = int(M['m01'] / (M['m00'] + 1e-5))  # calculate Y position
            
            # imageFrame = cv2.rectangle(imageFrame, (x, y),  
            #                            (x + w, y + h),  
            #                            (0, 0, 255), 2) 

            # Draw contours on output frame
            cv2.drawContours(imageFrame, contour, -1, BORDER_COLOUR, BORDER_THICKNESS)
            # Draw centre circle on output frame
            cv2.circle(imageFrame, (cx, cy), CENTRE_RADIUS, CENTRE_COLOUR, -1)
            # Put text on output frame
            cv2.putText(imageFrame, str(cx) + ',' + str(cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 1)

            # Identify if centre is over button
            if cx >= b1_x and cx <= b1_x + b1_w and cy >= b1_y and cy <= b1_y + b1_h:
                cv2.rectangle(imageFrame, (b1_x, b1_y), (b1_x + b1_w, b1_y + b1_h), (0, 255, 0), -1) 
            if cx >= b2_x and cx <= b2_x + b2_w and cy >= b2_y and cy <= b2_y + b2_h:
                cv2.rectangle(imageFrame, (b2_x, b2_y), (b2_x + b2_w, b2_y + b2_h), (0, 0, 255), -1)
  
    # # Creating contour to track green color 
    # contours, hierarchy = cv2.findContours(green_mask, 
    #                                        cv2.RETR_TREE, 
    #                                        cv2.CHAIN_APPROX_SIMPLE) 
      
    # for pic, contour in enumerate(contours): 
    #     area = cv2.contourArea(contour) 
    #     if(area > 300): 
    #         x, y, w, h = cv2.boundingRect(contour) 
    #         imageFrame = cv2.rectangle(imageFrame, (x, y),  
    #                                    (x + w, y + h), 
    #                                    (0, 255, 0), 2) 
              
    #         cv2.putText(imageFrame, "Green Colour", (x, y), 
    #                     cv2.FONT_HERSHEY_SIMPLEX,  
    #                     1.0, (0, 255, 0)) 
  
    # # Creating contour to track blue color 
    # contours, hierarchy = cv2.findContours(blue_mask, 
    #                                        cv2.RETR_TREE, 
    #                                        cv2.CHAIN_APPROX_SIMPLE) 
    # for pic, contour in enumerate(contours): 
    #     area = cv2.contourArea(contour) 
    #     if(area > 300): 
    #         x, y, w, h = cv2.boundingRect(contour) 
    #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
    #                                    (x + w, y + h), 
    #                                    (255, 0, 0), 2) 
              
    #         cv2.putText(imageFrame, "Blue Colour", (x, y), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 
    #                     1.0, (255, 0, 0))
       
    # Program Termination 
    cv2.imshow("Color Detection in Real-Time", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        webcam.release() 
        cv2.destroyAllWindows() 
        break
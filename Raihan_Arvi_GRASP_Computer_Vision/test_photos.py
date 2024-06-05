import cv2
from raihanarvi_graspcv import display_image

"""
This program is meant to test raihanarvi_graspcv.py
on the images taken of the competition images

Slight modifications have been made to raihan's code 
to make this modular

At the moment, the grip classification does not appear 
on the first frame, but pressing any key will display it.
However, this creates strange overlaps in the bounding boxes

Keep in the test branch
"""

photo = "image/pics_357.jpg"
is_video = 0  # goes into waitKey() (quick fix)
image = cv2.imread(photo)
display_image(image, is_video)

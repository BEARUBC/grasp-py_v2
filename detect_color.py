import numpy as np
import cv2

# image = cv2.imread("data/images/test/0a5e8d559aa5d680.jpg")

vid = cv2.VideoCapture(0)

(lower, upper) = ([150, 0, 0], [255, 150, 90])

lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

print(lower, upper)

active = True

while active:

    ret, image = vid.read()

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", np.hstack([image, output]))
    k = cv2.waitKey(1)
    if k == 27:
        break

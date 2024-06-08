import imutils
import numpy as np
from ultralytics import YOLO
import cv2
import colorspacious as cs

# Parameters

# Bounds for Blue
lower_limit_LAB = np.array([0, 0, 0])
upper_limit_LAB = np.array([1, 1, 1])

# Bounds for Red
# lower_limit_LAB = np.array([20, 150, 150])
# upper_limit_LAB = np.array([200, 255, 255])

THRESHOLD = 100  # Contours threshold

# Load a model
model = YOLO("./model/yolov8n.pt")  # load a custom model

# List of Acceptable Object
objects = [1, 2, 3]
values = ["power", "pinch", "hook"]
# Hashmap
hashmap = dict(zip(objects, values))

# GRIP TYPES ID ASSIGNMENT
# For now: bottle: 39 suitcase: 28 sandwich: 48 remote: 65
classes = [65, 39, 28, 48]
grips = ["power", ["pinch", "power"], "hook", "pinch"]
grip_types_assignments = dict(zip(classes, grips))

# Object(s) ID(s) that have multiple grips assigned
multiple_grips_objects = [39]

DEFAULT_GRIP = "power"

threshold = 0.5

# Camera
video = cv2.VideoCapture(1)
fullscreenState = False

# Program State
CALIBRATED = False
REFERENCE_OBJECT = 0

# Calibration
BORDER_COLOUR = (255, 0, 0)
BORDER_THICKNESS = 3
CENTRE_COLOUR = (0, 255, 0)
CENTRE_RADIUS = 5
CAL_RADIUS = 5
CAL_THICKNESS = 10
CAL_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
TEXT_SCALE = 0.5
THRESHOLD = 100 # can be adjusted to determine the distance at which we should identify objects

L_TOL = 80
A_TOL = 10
B_TOL = 10

# Default r based on previous methods
r = np.array([int((190 - 20)/2) + 20, int((255 - 150)/2) + 150, int((255 - 150)/2) + 150])

# Functions
def colour_limits(c, l_tol, a_tol, b_tol):
    lower_limit = np.array([max(0, c[0] - l_tol), max(1, c[1] - a_tol), max(1, c[2] - b_tol)], np.uint8)
    upper_imit = np.array([min(255, c[0] + l_tol), min(255, c[1] + a_tol), min(255, c[2] + b_tol)], np.uint8)

    return lower_limit, upper_imit

# IN: ListOfObjects
# OUT: Object
def getSingleHighestConfidenceObject(resultsList):
    def find_highest_index(scoresList):
        if not scoresList:
            raise ValueError("The list is empty")

        highest_number = max(scoresList)
        highest_index = scoresList.index(highest_number)

        return highest_index

    scoresList = []
    for result in resultsList:
        x1, y1, x2, y2, score, class_id = result
        scoresList.append(float(score))

    selected_object_index = find_highest_index(scoresList)
    return resultsList[selected_object_index]

# object = x1, y1, x2, y2, score, class_id
def isDesiredObjectType(detected_object, frame):
    is_matching_object = False

    if detected_object[5] == REFERENCE_OBJECT:
        is_matching_object = True
    else:
        is_matching_object = False

    is_matching_color = isColor(detected_object, frame)

    return is_matching_object and is_matching_color

def isColor(object, frame):
    coordinate = object[0:4]
    average_color = calculateColorComposition(coordinate, frame)

    def rgb_to_lab(rgb):
        # Normalize the RGB values to 0-1 range
        rgb_normalized = [x / 255.0 for x in rgb]

        # Convert RGB to LAB
        lab = cs.cspace_convert(rgb_normalized, "sRGB1", "CIELab")

        return lab

    lab_average_color = rgb_to_lab(average_color)

    def is_in_range(color):
        return (
            (upper_limit_LAB[0] > color[0] > lower_limit_LAB[0])
            and (upper_limit_LAB[1] > color[1] > lower_limit_LAB[1])
            and (upper_limit_LAB[2] > color[2] > lower_limit_LAB[2])
        )

    print(is_in_range(lab_average_color))
    return is_in_range(lab_average_color)  # return true if blue is in range

def calculateColorComposition(coordinate, frame):  # Works Well
    temporary_frame = frame.copy()
    # Blur the image for processing
    # frame_process = cv.medianBlur(frame, 3)
    frame_process = cv2.cvtColor(temporary_frame, cv2.COLOR_BGR2Lab)

    masked_color = cv2.inRange(frame_process, lower_limit_LAB, upper_limit_LAB)
    kernel = np.ones((5, 5), "uint8")
    masked_color = cv2.dilate(masked_color, kernel)
    # Creating contour to track selected color
    contours = cv2.findContours(masked_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    bounding_boxes_contour = []
    for c in contours:
        if cv2.contourArea(c) > THRESHOLD:
            x, y, w, h = cv2.boundingRect(c)
            boundingBox = [x, y, x + w, y + h]
            bounding_boxes_contour.append(boundingBox)

            M = cv2.moments(c)
            cx = int(
                M["m10"] / (M["m00"] + 1e-5)
            )  # calculate X position and add 1e-5 to avoid division by 0
            cy = int(M["m01"] / (M["m00"] + 1e-5))  # calculate Y position

            # Draw contours on output frame
            cv2.drawContours(frame, c, -1, (128, 255, 250), 3)
            # Draw centre circle on output frame
            cv2.circle(frame, (cx, cy), 5, (128, 255, 250), -1)
            # Put text on output frame
            cv2.putText(
                frame,
                str(cx) + "," + str(cy),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (128, 255, 250),
                1,
            )

    color_averages = []
    image_rgb = cv2.cvtColor(temporary_frame, cv2.COLOR_BGR2RGB)
    for b in bounding_boxes_contour:
        color_average = np.average(image_rgb[b[1] : b[3], b[0] : b[2]], axis=(0, 1))
        color_averages.append(color_average)

    # Averaging Contours Color
    red = 0
    green = 0
    blue = 0
    for color in color_averages:
        red = red + color[0]
        green = green + color[1]
        blue = blue + color[2]

    average_color = [
        red / (len(contours) + 1e-5),
        green / (len(contours) + 1e-5),
        blue / (len(contours) + 1e-5),
    ]

    return average_color



# Main
while True:
    ret, frame = video.read()
    if not ret:
        break
    # frame = cv2.resize(ret, 640, 480)
    fresh_frame = frame.copy()

    results = model(frame)[0]
    (h, w) = frame.shape[:2]

    rawResults = results.boxes.data.tolist()

    grip_type = ""
    if rawResults:
        highest_confidence_object = getSingleHighestConfidenceObject(rawResults)
        if highest_confidence_object:
            cv2.rectangle(frame, (int(highest_confidence_object[0]),
                                  int(highest_confidence_object[1])),
                          (int(highest_confidence_object[2]),
                           int(highest_confidence_object[3])), (0, 0, 255), 10)

        # Calibration Algorithm
        cv2.putText(frame, "Press 'c' to calibrate!",
                    (int(w / 2), int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 2)

        cal_x = int(highest_confidence_object[0]+(highest_confidence_object[2]-highest_confidence_object[0])/2)
        cal_y = int(highest_confidence_object[1]+(highest_confidence_object[3]-highest_confidence_object[1])/2)
        cv2.circle(frame, (cal_x, cal_y), CAL_RADIUS, CAL_COLOR, CAL_THICKNESS)
        # Convert the imageFrame in
        # BGR(RGB color space) to
        # LAB color space
        blurFrame = cv2.bilateralFilter(fresh_frame, 10, 100, 100)
        labFrame = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2LAB)
        if cv2.waitKey(1) == 99:  # press c
            r = labFrame[cal_y, cal_x]
            REFERENCE_OBJECT = highest_confidence_object

            # Set range for SELECTED color
            lower_limit_LAB, upper_limit_LAB = colour_limits(r, L_TOL, A_TOL, B_TOL)

        # Checking Object
        this_is_the_object_we_are_looking_for = isDesiredObjectType(highest_confidence_object, frame)


    # Display Grip Type
    # if grip_type is not None:
    #     cv2.putText(
    #         frame,
    #         "GRIP TYPE: " + grip_type,
    #         (0, 110),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         1.5,
    #         (255, 255, 255),
    #         3,
    #     )
    # else:
    #     cv2.putText(
    #         frame,
    #         "GRIP TYPE: " + DEFAULT_GRIP,
    #         (0, 110),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         1.5,
    #         (255, 255, 255),
    #         3,
    #     )

    # Draw Bounding Box
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(
                frame,
                results.names[int(class_id)].upper(),
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

    # Window Manager
    key = cv2.waitKey(1) & 0xFF
    cv2.namedWindow("GRASP Computer Vision", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("GRASP Computer Vision", cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GRASP Computer Vision", int(1920 / 2), int(1080 / 2))

    cv2.imshow("GRASP Computer Vision", frame)
    if key == 27:  # Escape
        cv2.destroyAllWindows()
        break

video.release()
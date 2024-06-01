import imutils
import numpy as np
from ultralytics import YOLO
import cv2
import colorspacious as cs

# Parameters

# Bounds for Blue
lower_limit_LAB = np.array([20, -200, -255])
upper_limit_LAB = np.array([200, 255, 100])

# Bounds for Red
# lower_limit_LAB = np.array([20, 150, 150])
# upper_limit_LAB = np.array([200, 255, 255])

THRESHOLD = 100 # Contours threshold

# Load a model
model = YOLO('./model/yolov8n.pt')  # load a custom model

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

# Functions

# IN: List
# OUT: List of Coordinates
def getDetectedObjectsCoordinate(resultsList):
    #coordinatesList = np.empty((0,), dtype=int)
    coordinatesList = []

    for result in resultsList:
        x1, y1, x2, y2, score, class_id = result

        #coordinate = np.array([int(x1), int(y1), int(x2), int(y2)])
        coordinate = [int(x1), int(y1), int(x2), int(y2)]
        #np.append(coordinatesList, coordinate)
        coordinatesList.append(coordinate)

    return coordinatesList

def getDetectedObjectsConfidenceScores(resultsList):
    scoresList = []

    for result in resultsList:
        x1, y1, x2, y2, score, class_id = result
        scoresList.append(float(score))

    return scoresList

#IN: scoresList
#OUT: class_id
def getHighestConfidenceObjectCoordinate(scoresList, resultsList):
    def find_highest_index(scoresList):
        if not scoresList:
            raise ValueError("The list is empty")

        highest_number = max(scoresList)
        highest_index = scoresList.index(highest_number)

        return highest_index

    highestIndex = find_highest_index(scoresList)
    selected_object = resultsList[highestIndex]

    coordinate_of_selected_object = selected_object[0:4]
    return coordinate_of_selected_object

#IN: scoresList
#OUT: string name of object
def getHighestConfidenceObjectName(scoresList, resultsList):
    def find_highest_index(scoresList):
        if not scoresList:
            raise ValueError("The list is empty")

        highest_number = max(scoresList)
        highest_index = scoresList.index(highest_number)

        return highest_index

    highestIndex = find_highest_index(scoresList)
    selected_object = resultsList[highestIndex]

    return selected_object[5]

# IN: CoordinateList out CoordinateList
# KMEANS AVERAGE COLOR CHECK
def getListOfObjectsWithBlue(coordinatesList, frame):
    coordinatesListBlue = []

    for coordinate in coordinatesList:
        if isBlue(coordinate, frame):
            coordinatesListBlue.append(coordinate)

    return coordinatesListBlue

def isBlue(coordinate, frame):
    average_color = calculateColorComposition(coordinate, frame)

    print(average_color)

    def rgb_to_lab(rgb):
        # Normalize the RGB values to 0-1 range
        rgb_normalized = [x / 255.0 for x in rgb]

        # Convert RGB to LAB
        lab = cs.cspace_convert(rgb_normalized, "sRGB1", "CIELab")

        return lab

    lab_average_color = rgb_to_lab(average_color)

    print(lab_average_color)

    def is_in_range(color):
        return (upper_limit_LAB[0] > color[0] > lower_limit_LAB[0]) and (upper_limit_LAB[1] > color[1] > lower_limit_LAB[1]) and (upper_limit_LAB[2] > color[2] > lower_limit_LAB[2])

    print(is_in_range(lab_average_color))

    return is_in_range(lab_average_color) # return true if blue is in range

def calculateColorComposition(coordinate, frame): # Works Well
    temporary_frame = frame.copy()

    # Blur the image for processing
    # frame_process = cv.medianBlur(frame, 3)

    frame_process = cv2.cvtColor(temporary_frame, cv2.COLOR_BGR2Lab)
    # frame_process = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    masked_color = cv2.inRange(frame_process, lower_limit_LAB, upper_limit_LAB)

    # Find contours:
    contours = cv2.findContours(masked_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    bounding_boxes_contour = []
    for c in contours:
        if cv2.contourArea(c) > THRESHOLD:
            x, y, w, h = cv2.boundingRect(c)
            boundingBox = [x, y, x+w, y+h]
            bounding_boxes_contour.append(boundingBox)

            M = cv2.moments(c)
            cx = int(M['m10'] / (M['m00'] + 1e-5))  # calculate X position and add 1e-5 to avoid division by 0
            cy = int(M['m01'] / (M['m00'] + 1e-5))  # calculate Y position

            # Draw contours on output frame
            cv2.drawContours(frame, c, -1, (128, 255, 250), 3)
            # Draw centre circle on output frame
            cv2.circle(frame, (cx, cy), 5, (128, 255, 250), -1)
            # Put text on output frame
            cv2.putText(frame, str(cx) + ',' + str(cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 250), 1)

    color_averages = []
    image_rgb = cv2.cvtColor(temporary_frame, cv2.COLOR_BGR2RGB)
    for b in bounding_boxes_contour:
        color_average = np.average(image_rgb[b[1]:b[3], b[0]:b[2]], axis=(0, 1))
        color_averages.append(color_average)

    # Averaging Contours Color
    red = 0
    green = 0
    blue = 0
    for color in color_averages:
        red = red + color[0]
        green = green + color[1]
        blue = blue + color[2]

    average_color = [red / (len(contours)+1e-5), green / (len(contours) + 1e-5), blue / (len(contours) + 1e-5)]

    print(average_color)

    return average_color

# IN: String
# OUT: Boolean
def isMultipleGripTypes(objectName): # Works
    return objectName in multiple_grips_objects

def determineGripTypesDependingOnWhichPartIsBlue(coordinateObject, frame): # Works
    upperHalfCoordinate = [coordinateObject[0], coordinateObject[1], coordinateObject[2], int(coordinateObject[3]/2)]
    lowerHalfCoordinate = [coordinateObject[0], int(coordinateObject[3]/2), coordinateObject[2], coordinateObject[3]]

    cv2.rectangle(frame, (int(coordinateObject[0]), int(coordinateObject[1])), (int(coordinateObject[2]), int(coordinateObject[3]/2)), (0, 255, 255), 6)
    cv2.rectangle(frame, (int(coordinateObject[0]), int(coordinateObject[3]/2)), (int(coordinateObject[2]), int(coordinateObject[3])), (0, 255, 255), 6)

    #print("Upper "+str(isBlue(upperHalfCoordinate, frame)))
    #print("Lower "+str(isBlue(lowerHalfCoordinate, frame)))

    if isBlue(upperHalfCoordinate, frame):
        #print("upper")
        return 0
    elif isBlue(lowerHalfCoordinate, frame):
        #print("lower")
        return 1

def assignGripType(objectClass, objectCoordinate, frame):
    objectClass = int(objectClass)
    grip_index = 5
    GRIP = ""

    if isMultipleGripTypes(objectClass):
        grip_index = determineGripTypesDependingOnWhichPartIsBlue(objectCoordinate, frame)
        #print(determineGripTypesDependingOnWhichPartIsBlue(objectCoordinate, frame))
        #STORED_GRIP = grip_types_assignments.get(objectClass)[grip_index]
        #GRIP = grip_types_assignments.get(objectClass)[grip_index]
        #print("Grips List : " + str(grip_types_assignments.get(objectClass)))
        #print("Selected : " + str(grip_types_assignments.get(objectClass)[grip_index]))
    else:
        STORED_GRIP = grip_types_assignments.get(objectClass)
        GRIP = grip_types_assignments.get(objectClass)

    return GRIP

# Main

while True:

    ret, frame = video.read()
    if not ret:
        break
    # frame = cv2.resize(ret, 640, 480)

    results = model(frame)[0]
    (h, w) = frame.shape[:2]

    # Check if Detected Objects are Acceptable, if yes, pick the first one.
    drawBox = False
    no_of_detected_objects = len(results.boxes.data.tolist())
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if int(class_id) in objects:
            drawBox = True
    #

    rawResults = results.boxes.data.tolist()
    coordinates = getDetectedObjectsCoordinate(rawResults)

    # Main Algorithm
    grip_type = ""
    if rawResults:
        detected_object_coordinates = getDetectedObjectsCoordinate(rawResults)
        detected_object_scores = getDetectedObjectsConfidenceScores(rawResults)
        detected_object_with_blue_coordinates = getListOfObjectsWithBlue(detected_object_coordinates, frame)
        print(detected_object_with_blue_coordinates)
        highest_confidence_blue_name = getHighestConfidenceObjectName(detected_object_scores, rawResults)
        highest_confidence_blue_coordinate = getHighestConfidenceObjectCoordinate(detected_object_scores, rawResults)

        grip_type = assignGripType(highest_confidence_blue_name, highest_confidence_blue_coordinate, frame)

    # Display Grip Type
    if grip_type is not None:
        cv2.putText(frame, "GRIP TYPE: " + grip_type, (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        cv2.putText(frame, "GRIP TYPE: " + DEFAULT_GRIP, (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Draw Bounding Box
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Window Manager
    key = cv2.waitKey(1) & 0xFF
    cv2.namedWindow('GRASP', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('GRASP', cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow('GRASP', int(1920 / 2), int(1080 / 2))

    cv2.imshow('GRASP Computer Vision', frame)
    if key == 27: # Escape
        cv2.destroyAllWindows()
        break

video.release()
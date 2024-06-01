from ultralytics import YOLO
import cv2


def detect_object():

    # get video
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()

    # get model
    model = YOLO("yolov8n.pt")
    threshold = 0.5
    results = model(frame)[0]

    # return list of detected objects
    return results.boxes.data.tolist()


if __name__ == "__main__":

    vid = cv2.VideoCapture(0)

    ret, frame = vid.read()

    model = YOLO("yolov8n.pt")

    threshold = 0.5

    while ret:

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4
                )
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
        cv2.imshow("video", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        ret, frame = vid.read()

    vid.release()
    cv2.destroyAllWindows()

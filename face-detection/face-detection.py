import cv2

import os

# Get the absolute path to the cascade file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
clf = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                flags=cv2.CASCADE_SCALE_IMAGE,
                                minSize=(30, 30))

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

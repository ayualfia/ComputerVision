import cv2

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
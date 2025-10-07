import cv2
import os

# Get the absolute path to the cascade file and set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
dataset_path = os.path.join(current_dir, "dataset/")
faceCascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(0)

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = 3 # id for person that we will detect
count = 0 # count for image name id
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1,
                                        minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count+=1
        cv2.imwrite(dataset_path+"Person-"+str(person_id)
                    +"-"+str(count)+".jpg", gray[y:y+h, x:x+w])

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 30: # stop when 30 photos have been taken
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import json

# ==== Paths ====
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
frontal_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
profile_path = os.path.join(base_dir, "haarcascade_profileface.xml")  # opsional; aman jika belum ada
model_path   = os.path.join(current_dir, "face-model.yml")
labels_path  = os.path.join(current_dir, "labels.json")

# ==== Init ====
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read(model_path)

frontal = cv2.CascadeClassifier(frontal_path)
profile = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None
if profile is None:
    print("[WARN] haarcascade_profileface.xml tidak ditemukan. Deteksi profile akan dilewati.")
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ['None', 'Alfi', 'Ervina']
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 100:
            id = names[id]
        else:
            name = "Unknown"
            color = (0, 0, 255)
            conf_txt = f">{THRESHOLD:.0f}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x+5, y-5), font, 1, color, 2)
        cv2.putText(frame, f"conf:{conf_txt}", (x+5, y+h-5), font, 0.8, color, 2)

    cv2.imshow("Camera (Robust Recognition)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

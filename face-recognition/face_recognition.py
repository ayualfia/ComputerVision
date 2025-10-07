import cv2
import os
import json

# ========== Paths ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
frontal_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
profile_path = os.path.join(base_dir, "haarcascade_profileface.xml")  # opsional
model_path   = os.path.join(current_dir, "face-model.yml")
labels_path  = os.path.join(current_dir, "labels.json")

# ========== Init ==========
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read(model_path)
frontal = cv2.CascadeClassifier(frontal_path)
profile = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None
if profile is None:
    print("[WARN] haarcascade_profileface.xml tidak ditemukan. Deteksi profile akan dilewati.")
font = cv2.FONT_HERSHEY_COMPLEX

# labels.json (opsional)
id_to_name = {}
if os.path.exists(labels_path):
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            id_to_name = json.load(f)
    except Exception as e:
        print(f"[WARN] Gagal baca labels.json: {e}. Pakai ID numerik saja.")

# Threshold LBPH (lebih kecil = lebih yakin). Sesuaikan 60–80
THRESHOLD = 70.0

# ========== Helpers ==========
def preprocess(gray_roi):
    roi = cv2.resize(gray_roi, (200, 200))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(roi)

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / float(area_a + area_b - inter)

def _no_big_overlap(box, boxes, thr=0.35):
    return all(_iou(box, b) <= thr for b in boxes)

def detect_faces_robust(gray):
    faces = list(frontal.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)))

    if profile is not None:
        for b in profile.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)):
            if _no_big_overlap(b, faces, 0.35):
                faces.append(b)

        flipped = cv2.flip(gray, 1)
        h, w = gray.shape
        for (x, y, ww, hh) in profile.detectMultiScale(flipped, 1.1, 5, minSize=(30, 30)):
            x_orig = w - x - ww
            b = (x_orig, y, ww, hh)
            if _no_big_overlap(b, faces, 0.35):
                faces.append(b)

    return faces

# ========== Run ==========
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_robust(gray)

    for (x, y, w, h) in faces:
        roi = preprocess(gray[y:y+h, x:x+w])
        label, conf = recognizer.predict(roi)  # LBPH: semakin kecil semakin yakin

        if conf <= THRESHOLD:
            name = id_to_name.get(str(label), f"ID {label}")
            color = (0, 255, 0)
            conf_txt = f"{conf:.1f}"
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

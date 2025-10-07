import cv2
import numpy as np
import os

# ==== Paths (gaya temenmu) ====
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
dataset_path = os.path.join(current_dir, "dataset")
model_path = os.path.join(current_dir, "face-model.yml")

# ==== Init ====
faceCascade = cv2.CascadeClassifier(cascade_path)

def checkDataset(directory=dataset_path):
    return os.path.exists(directory) and len(os.listdir(directory)) != 0

def preprocess(gray_roi):
    """Samakan ukuran + tingkatkan kontras biar stabil."""
    roi = cv2.resize(gray_roi, (200, 200))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(roi)

def organizeDataset(path=dataset_path):
    """
    Baca file 'dataset/Person-<id>-<n>.jpg', deteksi wajah (fallback: full),
    preprocessing, lalu AUGMENTASI (rotasi ±10°, ±20° + flip).
    """
    image_files = [os.path.join(path, p) for p in os.listdir(path)
                   if p.lower().endswith((".jpg", ".jpeg", ".png"))]

    faces, ids = [], np.array([], dtype="int")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ID dari nama file: Person-<id>-<n>.jpg
        filename = os.path.basename(img_path)
        try:
            pid = int(filename.split("-")[1])
        except Exception:
            print(f"[WARN] Lewati (format nama tidak sesuai): {filename}")
            continue

        # Deteksi wajah → ambil ROI terbesar; kalau kosong pakai full
        dets = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(dets) == 0:
            roi0 = gray
        else:
            x, y, w, h = max(dets, key=lambda b: b[2] * b[3])
            roi0 = gray[y:y+h, x:x+w]

        roi0 = preprocess(roi0)

        # -------- AUGMENTASI: rotasi ±10°, ±20° + flip --------
        aug_list = [roi0]
        h, w = roi0.shape
        for ang in (-20, -10, 10, 20):
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            aug_list.append(cv2.warpAffine(roi0, M, (w, h), borderMode=cv2.BORDER_REFLECT))
        aug_list.append(cv2.flip(roi0, 1))
        # -------------------------------------------------------

        for a in aug_list:
            faces.append(a)
            ids = np.append(ids, pid)

    return faces, ids

if not checkDataset():
    print("[ERROR] Dataset not found atau kosong. Jalankan face_create_dataset.py dulu.")
else:
    # Tuning LBPH untuk hasil lebih informatif & stabil
    recognizer = cv2.face.LBPHFaceRecognizer.create(radius=2, neighbors=8, grid_x=8, grid_y=8)

    print("[INFO] Training faces (dengan augmentasi)...")
    faces, ids = organizeDataset()
    if len(faces) == 0:
        print("[ERROR] Tidak ada sampel valid untuk training.")
    else:
        recognizer.train(faces, ids)
        recognizer.write(model_path)
        print(f"[OK] Training selesai! Samples: {len(faces)} | Unique IDs: {len(set(ids.tolist()))}")
        print(f"[OK] Model disimpan di '{model_path}'")

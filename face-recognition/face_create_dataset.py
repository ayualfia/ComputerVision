import cv2
import os
import json
import re

# ========== Paths ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
frontal_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
profile_path = os.path.join(base_dir, "haarcascade_profileface.xml")  # opsional
dataset_path = os.path.join(current_dir, "dataset/")
labels_path  = os.path.join(current_dir, "labels.json")

frontal = cv2.CascadeClassifier(frontal_path)
profile = cv2.CascadeClassifier(profile_path) if os.path.exists(profile_path) else None
if profile is None:
    print("[WARN] haarcascade_profileface.xml tidak ditemukan. Deteksi profile akan dilewati.")

# ========== Helpers ==========
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
    """
    1) Deteksi frontal
    2) Deteksi profile kiri
    3) Deteksi profile kanan (pakai flip)
    Tambahkan box hanya jika tidak overlap besar (IoU) dengan box yang sudah ada.
    """
    faces = list(frontal.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)))

    if profile is not None:
        # profile kiri
        for b in profile.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)):
            if _no_big_overlap(b, faces, 0.35):
                faces.append(b)

        # profile kanan via flip
        flipped = cv2.flip(gray, 1)
        h, w = gray.shape
        for (x, y, ww, hh) in profile.detectMultiScale(flipped, 1.1, 5, minSize=(30, 30)):
            x_orig = w - x - ww
            b = (x_orig, y, ww, hh)
            if _no_big_overlap(b, faces, 0.35):
                faces.append(b)

    return faces

def next_index_for_id(pid: int) -> int:
    os.makedirs(dataset_path, exist_ok=True)
    pat = re.compile(rf"^Person-{pid}-(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    max_idx = 0
    for fname in os.listdir(dataset_path):
        m = pat.match(fname)
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except:
                pass
    return max_idx + 1 if max_idx > 0 else 1

# ========== Nama -> ID (labels.json) ==========
name = input("Masukkan nama orang yang mau direkam: ").strip()
if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = {}

if name in labels.values():
    person_id = int([k for k, v in labels.items() if v == name][0])
    print(f"[INFO] Nama '{name}' sudah ada, pakai ID {person_id}")
else:
    person_id = max([int(k) for k in labels.keys()], default=0) + 1
    labels[str(person_id)] = name
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Tambah '{name}' sebagai ID {person_id}")

# ========== Capture ==========
cap = cv2.VideoCapture(0)
start_idx = next_index_for_id(person_id)
target = 30
count = 0

print(f"[INFO] Mulai ambil foto untuk {name} (ID {person_id}). Target: {target}. Tekan 'q' untuk selesai.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Kamera tidak terbaca.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_robust(gray)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        filename = os.path.join(dataset_path, f"Person-{person_id}-{start_idx + count}.jpg")
        cv2.imwrite(filename, roi)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} (ID {person_id}) - {count}/{target}",
                    (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if count >= target:
            break

    cv2.imshow("Camera (Robust Capture)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= target:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Tersimpan {count} foto untuk '{name}' (ID {person_id}) di folder dataset/")
print(f"[INFO] Mapping ID→Nama tersimpan di '{labels_path}'")

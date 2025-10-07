import cv2
import numpy as np
import os

def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "dataset")

def checkDataset(directory=None):
    if directory is None:
        directory = get_dataset_path()
    return os.path.exists(directory) and len(os.listdir(directory)) != 0

def preprocess(gray_roi):
    """samakan preprocessing dengan recognition"""
    gray_roi = cv2.resize(gray_roi, (200, 200))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_roi)

def organizeDataset(path=None):
    if path is None:
        path = get_dataset_path()
    imagePath = [os.path.join(path, p) for p in os.listdir(path)
                 if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    faces = []
    ids = np.array([], dtype="int")
    for i in imagePath:
        img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
        filename = os.path.basename(i)
        id = int(filename.split("-")[1])
        face = faceCascade.detectMultiScale(img, 1.1, 5, minSize=(30,30))
        if len(face) == 0:
            roi = img
        else:
            x,y,w,h = max(face, key=lambda b: b[2]*b[3])
            roi = img[y:y+h, x:x+w]
        roi = preprocess(roi)  # tambahkan ini biar konsisten
        faces.append(roi)
        ids = np.append(ids, id)
    return faces, ids

if not checkDataset():
    print("Dataset not found")
else:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
    
    recognizer = cv2.face.LBPHFaceRecognizer.create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    faceCascade = cv2.CascadeClassifier(cascade_path)

    print("Training faces...")
    faces, ids = organizeDataset()
    recognizer.train(faces, ids)
    print("Training finished!")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face-model.yml')
    recognizer.write(model_path)
    print(f"Model saved as '{model_path}'")

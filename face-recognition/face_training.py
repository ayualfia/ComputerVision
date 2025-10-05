import cv2
import numpy as np
import os

def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "dataset")

def checkDataset(directory=None):
    if directory is None:
        directory = get_dataset_path()
    if os.path.exists(directory) and len(os.listdir(directory)) != 0:
        return True
    return False

def organizeDataset(path=None):
    if path is None:
        path = get_dataset_path()
    imagePath = [os.path.join(path, p) for p in os.listdir(path)]
    faces = []
    ids = np.array([], dtype="int")
    for i in imagePath:
        img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
        # Use os.path.basename to get just the filename, regardless of OS path separators
        filename = os.path.basename(i)
        id = int(filename.split("-")[1])  # Get the ID number between 'Person-' and '-number.jpg'
        face = faceCascade.detectMultiScale(img)
        for (x, y, w, h) in face:
            faces.append(img[y:y+h, x:x+w])
            ids = np.append(ids, id)
    return faces, ids

if not checkDataset():
    print("Dataset not found")
else:
    # Get the absolute path to the cascade file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
    
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # train faces
    print("Training faces...")
    faces, ids = organizeDataset()
    recognizer.train(faces, ids)
    print("Training finished!")

    # save model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face-model.yml')
    recognizer.write(model_path)
    print(f"Model saved as '{model_path}'")

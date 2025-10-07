import cv2
import numpy as np
import os

def preprocess_face(image):
    # Consistent size for all faces
    desired_size = (200, 200)
    
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    image = cv2.equalizeHist(image)
    
    # Denoise the image
    image = cv2.fastNlMeansDenoising(image)
    
    # Resize to consistent dimensions
    image = cv2.resize(image, desired_size)
    
    # Normalize pixel values
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    return image

def detect_and_align_face(image, face_cascade):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    (x, y, w, h) = faces[0]  # Use the first face found
    face = image[y:y+h, x:x+w]
    face_gray = gray[y:y+h, x:x+w]
    
    # Eye detection for alignment
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(face_gray)
    
    if len(eyes) >= 2:
        # Sort eyes by x-coordinate to get left and right eye
        eyes = sorted(eyes, key=lambda x: x[0])
        left_eye, right_eye = eyes[:2]
        
        # Calculate angle for alignment
        left_eye_center = (int(left_eye[0] + left_eye[2]/2), int(left_eye[1] + left_eye[3]/2))
        right_eye_center = (int(right_eye[0] + right_eye[2]/2), int(right_eye[1] + right_eye[3]/2))
        
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image to align eyes horizontally
        center = (int(w/2), int(h/2))
        rotation_matrix = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        face = cv2.warpAffine(face, rotation_matrix, (w, h))
    
    return face

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
    skipped = 0
    
    for i in imagePath:
        img = cv2.imread(i)
        if img is None:
            print(f"Warning: Could not read image {i}")
            continue
            
        # Use os.path.basename to get just the filename, regardless of OS path separators
        filename = os.path.basename(i)
        try:
            id = int(filename.split("-")[1])  # Get the ID number between 'Person-' and '-number.jpg'
        except (IndexError, ValueError):
            print(f"Warning: Invalid filename format: {filename}")
            continue
        
        # Detect and align face
        face = detect_and_align_face(img, faceCascade)
        if face is not None:
            # Apply preprocessing
            processed_face = preprocess_face(face)
            faces.append(processed_face)
            ids = np.append(ids, id)
        else:
            skipped += 1
    
    print(f"Processed {len(faces)} faces. Skipped {skipped} images where no face was detected.")
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

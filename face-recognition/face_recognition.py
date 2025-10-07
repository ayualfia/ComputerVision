import cv2
import os
import numpy as np

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
        return None, None
    
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
    
    return face, (x, y, w, h)

# Get the absolute paths to the required files
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
model_path = os.path.join(current_dir, "face-model.yml")

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read(model_path) # face model from face_training.py
faceCascade = cv2.CascadeClassifier(cascade_path)
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ['None', 'Alfi', 'Ervina', 'guwa']
cap = cv2.VideoCapture(0)

min_confidence_threshold = 50  # Adjust this threshold as needed

while True:
    _, frame = cap.read()
    
    # Detect and align face
    face, face_rect = detect_and_align_face(frame, faceCascade)
    
    if face is not None:
        x, y, w, h = face_rect
        # Preprocess the face
        processed_face = preprocess_face(face)
        
        # Predict
        id, confidence = recognizer.predict(processed_face)
        confidence_score = 100 - confidence  # Convert to percentage
        
        # Draw rectangle around face
        color = (0, 255, 0) if confidence_score > min_confidence_threshold else (0, 0, 255)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        
        if confidence_score > min_confidence_threshold:
            id = names[id]
        else:
            id = "unknown"
        
        confidence = f"{confidence_score:.1f}%"

        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255,0,0), 1)
        cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1,
                    (255,255,0), 1)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def shear_image(image, shear_factor):
    height, width = image.shape[:2]
    shear_matrix = np.float32([
        [1, shear_factor, 0],
        [0, 1, 0]
    ])
    sheared = cv2.warpAffine(image, shear_matrix, (width, height))
    return sheared

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

def save_augmented_faces(image, base_filename, x, y, w, h, pose):
    # Original face
    face = image[y:y+h, x:x+w]
    cv2.imwrite(f"{base_filename}_{pose}.jpg", face)
    
    # Rotated versions (-15, -10, +10, +15 degrees)
    angles = [-15, -10, 10, 15]
    for angle in angles:
        cv2.imwrite(f"{base_filename}_{pose}_rot_{angle}.jpg", rotate_image(face, angle))
    
    # Sheared versions (horizontal shear)
    shear_factors = [-0.2, -0.1, 0.1, 0.2]
    for factor in shear_factors:
        cv2.imwrite(f"{base_filename}_{pose}_shear_{factor}.jpg", shear_image(face, factor))
    
    # Brightness and contrast variations
    brightness_values = [-30, 30]  # Darker and brighter
    contrast_values = [0.7, 1.3]   # Less and more contrast
    
    for brightness in brightness_values:
        cv2.imwrite(f"{base_filename}_{pose}_bright_{brightness}.jpg", 
                   adjust_brightness_contrast(face, brightness=brightness))
    
    for contrast in contrast_values:
        cv2.imwrite(f"{base_filename}_{pose}_contrast_{contrast}.jpg", 
                   adjust_brightness_contrast(face, contrast=contrast))

# Get the absolute path to the cascade file and set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
dataset_path = os.path.join(current_dir, "dataset/")
faceCascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(0)

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = 3  # id for person that we will detect
count = 0  # count for original image id
images_per_pose = 5  # Number of base images per pose
poses = ['front', 'left', 'right']  # Different poses to capture
current_pose_index = 0
augmentations_per_image = 12  # 4 rotations + 4 shears + 2 brightness + 2 contrast

# Calculate total expected images
total_augmented_images = images_per_pose * len(poses) * (1 + augmentations_per_image)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1,
                                       minNeighbors=5, minSize=(30, 30))

    current_pose = poses[current_pose_index]
    images_this_pose = count % images_per_pose
    
    # Display instructions
    instruction_text = f"Position: {current_pose.upper()} - "
    if current_pose == 'front':
        instruction_text += "Look directly at camera"
    elif current_pose == 'left':
        instruction_text += "Turn head slightly left"
    else:
        instruction_text += "Turn head slightly right"
        
    cv2.putText(frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    progress_text = f"Capturing {images_this_pose + 1}/{images_per_pose} for {current_pose} pose"
    cv2.putText(frame, progress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press SPACE to capture, 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space key to capture
        for (x,y,w,h) in faces:
            count += 1
            # Create base filename for this capture
            base_filename = os.path.join(dataset_path, f"Person-{person_id}-{count}")
            
            # Save original and augmented versions
            save_augmented_faces(gray, base_filename, x, y, w, h, current_pose)
            
            print(f"Captured image set {count} for {current_pose} pose")
            
            # Move to next pose if we've captured enough images for current pose
            if count % images_per_pose == 0:
                current_pose_index = (current_pose_index + 1) % len(poses)
                if current_pose_index == 0:  # We've completed all poses
                    if count >= images_per_pose * len(poses):
                        print(f"\nDataset creation complete!")
                        print(f"Total base images: {count}")
                        print(f"Total augmented images: {count * (1 + augmentations_per_image)}")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

cap.release()
cv2.destroyAllWindows()

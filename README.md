# Computer Vision Project

This project implements various computer vision applications using OpenCV, including face detection, face recognition, and hand skeleton detection.

## Features

- **Face Detection**: Detect faces in real-time using webcam
- **Face Recognition**: Recognize and identify faces using LBPH Face Recognizer
- **Hand Skeleton Detection**: Detect hand skeleton in real-time
- **Dataset Creation**: Tools to create your own face dataset for training

## Project Structure

```
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── webcam-check.py
├── face-detection/
│   └── face-detection.py
├── face-recognition/
│   ├── face_create_dataset.py
│   ├── face_recognition.py
│   ├── face_training.py
│   ├── face-model.yml
│   └── dataset/
└── hand-skeleton/
    └── hand-skeleton.py
```

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

Install all required packages using:
```bash
pip install -r requirements.txt
```

## Setup and Installation

1. Clone this repository or download the source code
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Face Detection

Run the face detection script:
```bash
python face-detection/face-detection.py
```

### Face Recognition

1. First, create a dataset of faces:
   ```bash
   python face-recognition/face_create_dataset.py
   ```
   - Press 'q' to quit after capturing images
   - The script will capture 30 images automatically

2. Train the face recognizer:
   ```bash
   python face-recognition/face_training.py
   ```

3. Run face recognition:
   ```bash
   python face-recognition/face_recognition.py
   ```
   - Press 'q' to quit

### Hand Skeleton Detection

Run the hand skeleton detection script:
```bash
python hand-skeleton/hand-skeleton.py
```

## Key Controls

- Press 'q' to quit any of the applications
- For dataset creation, the program will automatically capture 30 images and stop

## Notes

- Make sure your webcam is properly connected and accessible
- Good lighting conditions will improve detection accuracy
- For face recognition, ensure you have created a dataset and trained the model before running recognition
- The face recognition model is saved as 'face-model.yml' after training

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are properly installed
2. Check if your webcam is working properly (use webcam-check.py)
3. Verify that the cascade classifier file (haarcascade_frontalface_default.xml) is in the correct location
4. For face recognition, ensure you have:
   - Created a dataset
   - Trained the model
   - Have the face-model.yml file in the correct location
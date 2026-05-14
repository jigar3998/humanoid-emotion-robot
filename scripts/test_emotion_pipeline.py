# scripts/test_emotion_pipeline.py
# Tests face detection + emotion classification on a webcam frame or test image.
# Run on Mac to verify pipeline before deploying to Nano.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                '../ros2_ws/src/emotion_pkg'))

import cv2
import numpy as np
from emotion_pkg.face_detector import FaceDetector
from emotion_pkg.emotion_classifier import EmotionClassifier

print('Loading models...')
detector   = FaceDetector(use_trt=False)
classifier = EmotionClassifier(use_trt=False)
print('Models loaded')

# Try webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('No webcam — using synthetic test image')
    frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
else:
    ret, frame = cap.read()
    cap.release()
    if not ret:
        frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    else:
        print(f'Captured webcam frame: {frame.shape}')

boxes = detector.detect(frame, conf_thresh=0.3)
print(f'Faces detected: {len(boxes)}')

for i, box in enumerate(boxes):
    face = detector.align_face(frame, box)
    emotion, confidence = classifier.predict(face)
    print(f'  Face {i+1}: {emotion} ({confidence:.2%})')

if not boxes:
    print('No faces in frame — testing classifier directly with random crop')
    face = np.random.randint(80, 180, (260, 260, 3), dtype=np.uint8)
    emotion, confidence = classifier.predict(face)
    print(f'  Classifier output: {emotion} ({confidence:.2%})')

print('Pipeline test complete')

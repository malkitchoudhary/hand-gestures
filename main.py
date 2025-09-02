import cv2
import mediapipe as mp
import pyautogui

# Mediapipe face detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Camera capture
cap = cv2.VideoCapture(0)

# Face detection confidence threshold
SHORTS_CONFIDENCE_THRESHOLD = 0.8

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=SHORTS_CONFIDENCE_THRESHOLD) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert the image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process the image
        results = face_detection.process(image)

        # Convert back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If no face is detected, we assume it's a Shorts and skip
        if not results.detections:
            print("No face detected - Skipping (like Shorts)")
            pyautogui.press("right")  # Simulates pressing right arrow to skip
        else:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Show the image
        cv2.imshow('YouTube Shorts Skipper', image)

        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import tensorflow as tf

# # Load model
# model = tf.keras.models.load_model('model/hand_gesture_model.h5')

# # Emojis for class names: A = Palm, B = Fist, C = Thumbs Up
# class_names = ['Palm', ' Fist', ' Thumbs Up']

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Region of interest (ROI) box
#     x1, y1, x2, y2 = 100, 100, 300, 300
#     roi = frame[y1:y2, x1:x2]
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Preprocess ROI
#     roi_resized = cv2.resize(roi, (64, 64))
#     roi_normalized = roi_resized.astype("float32") / 255.0
#     roi_input = np.expand_dims(roi_normalized, axis=0)

#     # Predict gesture
#     prediction = model.predict(roi_input)
#     predicted_class = np.argmax(prediction)
    
#     # 🟡 EMOJI LABEL HERE:
#     label = class_names[predicted_class]
#     confidence = prediction[0][predicted_class] * 100

#     # 🟡 DISPLAY LABEL + EMOJI + CONFIDENCE:
#     cv2.putText(frame, f"{label} ({confidence:.2f}%)", (100, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

#     # Show the frame
#     cv2.imshow("Hand Gesture Detection", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

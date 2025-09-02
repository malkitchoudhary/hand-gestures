# detect_ui.py
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import tkinter as tk
import os

# Load model and class names
model = tf.keras.models.load_model("model/gesture_model.h5")
class_names = ['Palm', 'Fist', 'Thumbs Up']
img_size = 224

# UI Setup
window = tk.Tk()
window.title("Gesture Detection")
window.geometry("800x600")
window.configure(bg="#202020")

# Webcam Display
video_label = tk.Label(window)
video_label.pack(pady=10)

# Prediction Label
gesture_text = tk.Label(window, text="Waiting...", font=("Helvetica", 20), fg="white", bg="#202020")
gesture_text.pack(pady=10)

# Confidence Label
confidence_label = tk.Label(window, text="", font=("Helvetica", 14), fg="lightgray", bg="#202020")
confidence_label.pack()

# Load Webcam
cap = cv2.VideoCapture(0)

# Prediction Stability
last_prediction = ""
stable_count = 0
STABILITY_THRESHOLD = 5

def update_frame():
    global last_prediction, stable_count

    ret, frame = cap.read()
    if not ret:
        return

    # Flip + ROI box
    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 150, 100, 150 + 224, 100 + 224
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess
    img = cv2.resize(roi, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx] * 100
    predicted_label = class_names[class_idx]

    # Stability logic
    if predicted_label == last_prediction:
        stable_count += 1
    else:
        stable_count = 0
    last_prediction = predicted_label

    if stable_count > STABILITY_THRESHOLD and confidence > 80:
        gesture_text.config(text=f"Gesture: {predicted_label}", fg="cyan")
        confidence_label.config(text=f"Confidence: {confidence:.1f}%")
    else:
        gesture_text.config(text="Detecting...", fg="white")
        confidence_label.config(text="")

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    window.after(10, update_frame)

# Start
update_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()

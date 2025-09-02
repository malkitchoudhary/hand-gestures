import cv2
import os

# Change this for each gesture
gesture_name = 'A'  # Change to 'B' and 'C' later
save_path = f'dataset/{gesture_name}'
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("[INFO] Press 's' to save image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)

    cv2.imshow("Collecting Images", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Save the image inside the ROI
        filename = f"{save_path}/{count}.jpg"
        cv2.imwrite(filename, roi)
        count += 1
        print(f"[INFO] Saved: {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

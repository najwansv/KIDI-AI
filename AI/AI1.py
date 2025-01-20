import cv2
import os
import torch

# Load the YOLOv5 model
model_path = '/Model/yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Open the RTSP stream
rtsp_url = 'rtsp://admin:telkomiot123@192.168.1.110:554/cam/realmonitor?channel=1&subtype=0'
cap = cv2.VideoCapture(rtsp_url)

# Create a named window
cv2.namedWindow('RTSP Stream', cv2.WINDOW_NORMAL)

# Resize the window to fit the screen
screen_width = 1920  # Replace with your screen width
screen_height = 1080  # Replace with your screen height
cv2.resizeWindow('RTSP Stream', screen_width, screen_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes on the frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow('RTSP Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Object detection on RTSP stream completed.")
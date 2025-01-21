import cv2
import torch
# from flask import Flask, Response

# Initialize Flask app
# app = Flask(__name__)

# Load the YOLOv5 model
model_path = '/Model/yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# RTSP stream URL
# rtsp_url = 'rtsp://admin:telkomiot123@36.92.168.180:11054/cam/realmonitor?channel=1&subtype=0'

def All_Obj_Detection(rtsp_url):import cv2
import torch

# Load the YOLOv5 model
model_path = '/Model/yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

def All_Obj_Detection_In_Boundary(rtsp_url):
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    # Define the boundary area (rectangle)
    boundary_x1, boundary_y1 = 100, 100  # top-left corner
    boundary_x2, boundary_y2 = 800, 1500  # bottom-right corner

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes on the frame
        annotated_frame = results.render()[0]

        # Create a writable copy of the annotated frame
        annotated_frame = annotated_frame.copy()

        # Count objects in the boundary area
        obj_count = 0
        for pred in results.pred[0]:
            # Get the coordinates of the bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, pred[:4])

            # Check if the bounding box intersects with the boundary area
            if x1 >= boundary_x1 and y1 >= boundary_y1 and x2 <= boundary_x2 and y2 <= boundary_y2:
                obj_count += 1

        # Draw the boundary box on the frame
        cv2.rectangle(annotated_frame, (boundary_x1, boundary_y1), (boundary_x2, boundary_y2), (255, 0, 0), 2)

        # Display object count in the boundary area
        cv2.putText(annotated_frame, f'Objects in boundary: {obj_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



    cap.release()

    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes on the frame
        annotated_frame = results.render()[0]
        # Draw bounding boxes on the frame
        annotated_frame = results.render()[0]

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

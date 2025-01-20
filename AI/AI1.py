import cv2
import torch
from flask import Flask, Response
import time

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv5 model
model_path = '/Model/yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# RTSP stream URL
rtsp_url = 'rtsp://admin:telkomiot123@192.168.1.110:554/cam/realmonitor?channel=1&subtype=1'

def generate_frames():
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes on the frame
        annotated_frame = results.render()[0]

        # Get frame dimensions
        height, width, _ = annotated_frame.shape

        # Add FPS to the frame (upper right corner)
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    # Route for video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
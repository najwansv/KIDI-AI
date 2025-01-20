import cv2
import torch
from flask import Flask, Response

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv5 model
model_path = '/Model/yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# RTSP stream URL
rtsp_url = 'rtsp://admin:telkomiot123@192.168.1.110:554/cam/realmonitor?channel=1&subtype=0'

def generate_frames():
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
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

import http.server
import socketserver
from threading import Thread
from flask import Flask, request, Response
from flask_cors import CORS
import cv2

# Specify the directory containing your web files
DIRECTORY = "Dashboard"
PORT = 80  # Port for serving the HTML

# Flask app for backend
app = Flask(__name__)
CORS(app)

# Global variables
streaming = False  # Flag to control streaming
rtsp_url = None  # RTSP URL

def generate_frames():
    global streaming, rtsp_url
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    while streaming and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global streaming, rtsp_url
    rtsp_url = request.form.get('rtsp')
    
    if not rtsp_url:
        return "RTSP link is required", 400

    if streaming:  # Check if streaming is already running
        return "Streaming is already running", 400

    streaming = True
    return "Streaming started", 200

@app.route('/stop_streaming', methods=['POST'])
def stop_streaming():
    global streaming
    if streaming:
        streaming = False
        return "Streaming stopped", 200
    return "No streaming process to stop", 400

@app.route('/video_feed')
def video_feed():
    if not streaming:
        return "Streaming is stopped", 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask server in a separate thread
def start_flask():
    app.run(port=5001)  # Port for Flask API

# Custom HTTP server handler
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

# Start the HTTP server
def start_http_server():
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving HTTP on localhost:{PORT} (http://127.0.0.1:{PORT})")
        httpd.serve_forever()

# Run both servers
if __name__ == "__main__":
    Thread(target=start_flask).start()
    start_http_server()
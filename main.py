# main.py
import http.server
import socketserver
from threading import Thread
from flask import Flask, Response, request
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
ai_mode = None  # AI mode
videos = {
    'noAI': 'AI1.py',  # Video for noAI mode
    'AI1': 'placeholder_AI.mp4',    # Video for AI1 mode
    'AI2': 'placeholder_AI2.mp4',    # Video for AI2 mode
    'AI3': 'placeholder_AI3.mp4',    # Video for AI3 mode
    'AI4': 'placeholder_AI4.mp4',    # Video for AI4 mode
}

# Function to stream video
def stream_video(video_path):
    """Stream video frames from a video file."""
    print(f"video path : {video_path}")
    cap = cv2.VideoCapture(video_path)

    while streaming and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global streaming, ai_mode
    # Get the AI mode from the POST request
    ai_mode = request.form.get('ai_mode')  # Default to 'noAI' if not provided
    print(f"Streaming selected mode: {ai_mode}")

    if streaming:  # Check if streaming is already running
        return "Streaming is already running", 400

    if ai_mode not in videos:
        return "Invalid AI mode", 400

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
    global ai_mode, streaming
    if not streaming:
        return "Streaming is stopped", 400
    
    # Ensure valid AI mode
    if ai_mode not in videos:
        return "Invalid AI mode", 400

    # Reset the video stream by forcing a re-initialization
    streaming = False  # Stop the current stream
    video_path = videos[ai_mode]
    streaming = True   # Start the new stream with the updated AI mode

    # Stream the appropriate video based on AI mode
    print(f"Streaming in mode: {ai_mode, video_path}")
    return Response(stream_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/update_ai_mode', methods=['POST'])
def update_ai_mode():
    global ai_mode
    new_ai_mode = request.form.get('ai_mode')
    if new_ai_mode not in videos:
        return "Invalid AI mode", 400
    
    ai_mode = new_ai_mode
    print(f"AI Mode updated to: {ai_mode}")
    
    # Call video_feed to reset the stream with the new AI mode
    return video_feed()


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

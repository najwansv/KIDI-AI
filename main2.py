# main.py
import http.server
import socketserver
from threading import Thread
from flask import Flask, request, Response
from flask_cors import CORS
from AI.NonAI import generate_frames  # Import the function
from AI.AI1 import All_Obj_Detection 
from AI.AI2 import All_Obj_Detection_In_Boundary
from AI.AI3 import Obj_Counter


# Specify the directory containing your web files
DIRECTORY = "Dashboard"
PORT = 80  # Port for serving the HTML

# Flask app for backend
app = Flask(__name__)
CORS(app)

# Global variables
streaming = False  # Flag to control streaming
rtsp_url = None  # RTSP URL
ai_mode = None  # AI mode

@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global streaming, rtsp_url, ai_mode
    rtsp_url = request.form.get('rtsp')
    ai_mode = request.form.get('ai_mode')

    if not rtsp_url:
        return "RTSP link is required", 400

    if streaming:  # Check if streaming is already running
        return "Streaming is already running", 400

    streaming = True

    # Start streaming with NonAI logic
    if not ai_mode or ai_mode not in ['AI1', 'AI2', 'AI3', 'AI4']:
        print("Running NonAI mode")
        Thread(target=lambda: generate_frames(rtsp_url)).start()

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
    global ai_mode

    streaming = False  # Stop the current stream
    streaming = True

    if not streaming:
        return "Streaming is stopped", 400
    if ai_mode == 'AI1':
        print("AI1")
        return Response(All_Obj_Detection(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI2':
        print("AI2")
        return Response(All_Obj_Detection_In_Boundary(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    if ai_mode == 'AI3':
        print("AI3")
        return Response(Obj_Counter(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

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
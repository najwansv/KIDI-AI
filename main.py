import http.server
import socketserver
from threading import Thread
from flask import Flask, request
from flask_cors import CORS
import subprocess

# Specify the directory containing your web files
DIRECTORY = "Dashboard"
PORT = 80  # Port for serving the HTML

# Flask app for backend
app = Flask(__name__)
CORS(app)

process = None  # Global variable to store the streaming process

# Route to start streaming
@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global process
    rtsp_link = request.form.get('rtsp')
    
    if not rtsp_link:
        return "RTSP link is required", 400

    if process and process.poll() is None:  # Check if a process is already running
        return "Streaming is already running", 400

    try:
        # Run the Python script with the RTSP link as an argument
        process = subprocess.Popen(['python', 'AI\\NonAI.py', rtsp_link])
        return "Streaming started", 200
    except Exception as e:
        return str(e), 500

# Route to stop streaming
@app.route('/stop_streaming', methods=['POST'])
def stop_streaming():
    global process
    if process:
        try:
            process.kill()  # Terminate the process
            process.wait()  # Ensure the process has fully stopped
            process = None
            return "Streaming stopped", 200
        except Exception as e:
            return str(e), 500
    return "No streaming process to stop", 400

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

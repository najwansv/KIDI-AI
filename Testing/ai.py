import os
from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

# Directory to store HLS files
HLS_DIR = 'hls'
os.makedirs(HLS_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    rtsp_url = request.form.get('rtsp_url')
    if not rtsp_url:
        return jsonify({'error': 'No RTSP URL provided'}), 400

    # Stop any existing FFmpeg processes
    stop_ffmpeg_process()

    # Generate HLS with FFmpeg
    hls_path = os.path.join(HLS_DIR, 'stream.m3u8')
    ffmpeg_command = [
        'ffmpeg',
        '-i', rtsp_url,  # Input RTSP URL
        '-c:v', 'libx264',  # Encode to H.264
        '-preset', 'ultrafast',
        '-f', 'hls',  # Output format: HLS
        '-hls_time', '1',  # HLS segment duration
        '-hls_list_size', '5',  # Number of HLS segments in the playlist
        '-hls_flags', 'delete_segments',  # Automatically delete old segments
        hls_path
    ]

    # Run FFmpeg process
    subprocess.Popen(ffmpeg_command)

    return jsonify({'message': 'Stream started successfully'})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    stop_ffmpeg_process()
    return jsonify({'message': 'Stream stopped successfully'})

def stop_ffmpeg_process():
    # Kill all FFmpeg processes
    subprocess.call(['pkill', '-f', 'ffmpeg'])

@app.route('/video_feed')
def video_feed():
    # Serve the HLS playlist
    return app.send_static_file(os.path.join(HLS_DIR, 'stream.m3u8'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

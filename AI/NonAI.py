import cv2
import argparse
from flask import Flask, Response

# Initialize Flask app
app = Flask(__name__)

def generate_frames(rtsp_url):
    # Open a connection to the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    # Use the globally defined RTSP URL
    return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream an RTSP video feed through Flask.")
    parser.add_argument("rtsp_url", type=str, help="The RTSP URL of the video stream.")
    args = parser.parse_args()

    # Make the RTSP URL available globally
    rtsp_url = args.rtsp_url

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

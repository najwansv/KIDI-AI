# from flask import Flask, Response
# import cv2
# import argparse

# app = Flask(__name__)

# # Global flag to control streaming
# streaming = True

# def generate_frames(rtsp_url):
#     global streaming
#     cap = cv2.VideoCapture(rtsp_url)

#     if not cap.isOpened():
#         print("Error: Could not open video stream")
#         return

#     while streaming and cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not read frame")
#             break

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     global streaming
#     streaming = True  # Reset the flag when starting the feed
#     return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/stop_streaming', methods=['POST'])
# def stop_streaming():
#     global streaming
#     streaming = False  # Set the flag to False to stop the feed
#     return "Streaming stopped", 200

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Stream an RTSP video feed through Flask.")
#     parser.add_argument("rtsp_url", type=str, help="The RTSP URL of the video stream.")
#     args = parser.parse_args()

#     # Make the RTSP URL available globally
#     rtsp_url = args.rtsp_url

#     # Run the Flask app
#     app.run(host='0.0.0.0', port=5000, debug=True)
import cv2
import torch

model_path = '/Model/yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Define a counting line (adjust as per your video resolution)
LINE_POSITION = 1200  # Vertical position of the line
object_counts = {}  # Dictionary to track counts of all detected object categories


def All_Obj_Detection(rtsp_url):
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

def get_centroid(box):
    """Calculate the centroid of a bounding box."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def Obj_Counter(rtsp_url):
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

        # Parse the results
        detections = results.xyxy[0].numpy()  # x1, y1, x2, y2, confidence, class

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            cls_name = model.names[int(cls)]  # Get class name
            cx, cy = get_centroid([x1, y1, x2, y2])

            # Initialize category count if not already in dictionary
            if cls_name not in object_counts:
                object_counts[cls_name] = 0

            # Check if the object crosses the line
            if LINE_POSITION - 5 < cy < LINE_POSITION + 5:  # Add some margin
                object_counts[cls_name] += 1
                print(f"{cls_name} crossed the line. Total: {object_counts[cls_name]}")

            # Draw bounding boxes and the centroid
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, cls_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the counting line
        cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 0, 255), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

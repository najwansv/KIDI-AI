import cv2
import torch
from deepface import DeepFace

model_path = '/Model/yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Define a counting line (adjust as per your video resolution)
LINE_POSITION = 1200  # Vertical position of the line
object_counts = {}  # Dictionary to track counts of all detected object categories
boundary_objects = {}  # Dictionary to track objects in the boundary area


#======================= Non AI ===========================================
def generate_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

#======================= AI 1 ===========================================
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

#======================= AI 2 ===========================================
def is_in_boundary(box, b_x1, b_y1, b_x2, b_y2):
    x1, y1, x2, y2 = map(int, box)
    return x1 >= b_x1 and y1 >= b_y1 and x2 <= b_x2 and y2 <= b_y2



def All_Obj_Detection_In_Boundary(rtsp_url):
    global boundary_objects
    boundary_objects = {}
    cap = cv2.VideoCapture(rtsp_url)
    boundary_x1, boundary_y1 = 100, 100
    boundary_x2, boundary_y2 = 500, 400
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results.render()[0].copy()

        boundary_objects = {}
        for pred in results.pred[0]:
            if is_in_boundary(pred[:4], boundary_x1, boundary_y1, boundary_x2, boundary_y2):
                cls_name = model.names[int(pred[5])]
                boundary_objects[cls_name] = boundary_objects.get(cls_name, 0) + 1

        # Draw boundary boxhttps://chatgpt.com/gpts
        cv2.rectangle(annotated_frame, (boundary_x1, boundary_y1), 
                     (boundary_x2, boundary_y2), (255, 0, 0), 2)

        # Display counts
        y_offset = 50
        for obj, count in boundary_objects.items():
            cv2.putText(annotated_frame, f'{obj}: {count}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 0, 0), 2)
            y_offset += 30

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


#======================= AI 3 ===========================================

def calculate_centroid(x1, y1, x2, y2):
    # Calculate the centroid of the bounding box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

def Obj_Counter(rtsp_url):
    global object_counts
    object_counts = {}  # Reset at start
    tracked_objects = {}  # Track objects across frames
    
    cap = cv2.VideoCapture(rtsp_url)
    line_x = 500  # Vertical line x-coordinate
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        width = frame.shape[1]
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 0), 2)

        # Run YOLOv5 model on the frame
        results = model(frame)
        annotated_frame = results.render()[0].copy()

        current_objects = {}

        for pred in results.pred[0]:
            x1, y1, x2, y2 = map(int, pred[:4])
            cx, cy = calculate_centroid(x1, y1, x2, y2)
            cls_name = model.names[int(pred[5])]  # Get object class name
            
            # Create a unique identifier for each object based on its class and position
            obj_id = f"{cls_name}_{x1}_{y1}_{x2}_{y2}"
            
            # Initialize object count if not exists
            if cls_name not in object_counts:
                object_counts[cls_name] = 0
            
            # Draw centroid on the object
            cv2.circle(annotated_frame, (cx, cy), 10, (0, 0, 255), -1)
            
            # Check if this object was tracked in previous frames
            if obj_id in tracked_objects:
                prev_state = tracked_objects[obj_id]
                # Count object only if it crosses the line from left to right or right to left
                if (prev_state['cx'] < line_x and cx >= line_x) or (prev_state['cx'] > line_x and cx <= line_x):
                    object_counts[cls_name] += 1
            
            # Update current tracked objects
            current_objects[obj_id] = {
                'cx': cx,
                'cy': cy,
                'cls_name': cls_name
            }
        
        # Update tracked objects for the next frame
        tracked_objects = current_objects

        # Display counts on the frame
        y_offset = 50
        for obj, count in object_counts.items():
            cv2.putText(annotated_frame, f'{obj}: {count}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            y_offset += 30

        # Yield the frame as a JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

#======================= AI 4 ===========================================
def Gender_Mood_Age_Detection(rtsp_url):
    """
    Detect gender, mood, and age from faces in a video stream.

    :param rtsp_url: URL of the RTSP video stream
    :yield: Frames with gender, mood, and age annotations
    """
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze the frame for face attributes
        try:
            analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        except Exception as e:
            print(f"Error during face analysis: {e}")
            continue

        # Annotate the frame with the detected information
        for face in analysis:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

            # Extract attributes
            gender = face['gender']
            age = face['age']
            emotion = max(face['emotion'], key=face['emotion'].get)  # Most dominant emotion

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display gender, age, and emotion
            info = f"Gender: {gender}, Age: {age}, Mood: {emotion}"
            cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

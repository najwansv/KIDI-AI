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

        # Draw boundary box
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

def crosses_line(cy, line_y=300):
    return abs(cy - line_y) < 5

def Obj_Counter(rtsp_url):
    global object_counts
    cap = cv2.VideoCapture(rtsp_url)
    line_y = 300  # Crossing line y-coordinate
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        height = frame.shape[0]
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

        results = model(frame)
        annotated_frame = results.render()[0].copy()

        for pred in results.pred[0]:
            x1, y1, x2, y2 = map(int, pred[:4])
            cy = (y1 + y2) // 2
            cls_name = model.names[int(pred[5])]
            
            if cls_name not in object_counts:
                object_counts[cls_name] = 0
                
            if crosses_line(cy, line_y):
                if cy < height//2:  # Only count objects moving down
                    object_counts[cls_name] += 1

        # Display counts
        y_offset = 50
        for obj, count in object_counts.items():
            cv2.putText(annotated_frame, f'{obj}: {count}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            y_offset += 30

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

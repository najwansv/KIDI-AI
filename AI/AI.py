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

def is_crossing_line(center, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    if y1 == y2:  # Horizontal line
        return center[1] == y1
    elif x1 == x2:  # Vertical line
        return center[0] == x1
    return False

def calculate_centroid(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

def Obj_Counter(rtsp_url):
    global object_counts
    object_counts = {}
    
    cap = cv2.VideoCapture(rtsp_url)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        height, width = frame.shape[:2]
        # Define counting line in the middle
        line = [(width // 2, 0), (width // 2, height)]
        
        # Draw the counting line
        cv2.line(frame, line[0], line[1], (0, 255, 0), 2)

        results = model(frame)
        annotated_frame = results.render()[0].copy()

        for pred in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, pred[:4])
            center = calculate_centroid(x1, y1, x2, y2)
            cls_name = model.names[int(pred[5])]
            
            # Initialize object count if not exists
            if cls_name not in object_counts:
                object_counts[cls_name] = 0
            
            # Draw centroid on the object
            cv2.circle(annotated_frame, center, 10, (0, 0, 255), -1)
            
            # Check if object is crossing the line
            if is_crossing_line(center, line):
                object_counts[cls_name] += 1
            
            # Draw bounding box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(annotated_frame, cls_name, (x1, y1 - 10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display counts on the frame
        y_offset = 50
        for obj, count in object_counts.items():
            cv2.putText(annotated_frame, f'{obj}: {count}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            y_offset += 30

        # Yield the frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

#======================= AI 4 ===========================================
def Gender_Mood_Age_Detection(rtsp_url):
    """
    Detect gender, mood, and age using Caffe models from video stream.
    
    :param rtsp_url: URL of the RTSP video stream
    :yield: Frames with gender, mood, and age annotations
    """
    # Load models
    faceProto = "Model/caffemodel/opencv_face_detector.pbtxt"
    faceModel = "Model/caffemodel/opencv_face_detector_uint8.pb"
    ageProto = "Model/caffemodel/age_deploy.prototxt"
    ageModel = "Model/caffemodel/age_net.caffemodel"
    genderProto = "Model/caffemodel/gender_deploy.prototxt"
    genderModel = "Model/caffemodel/gender_net.caffemodel"

    # Initialize networks
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # Model parameters
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    padding = 20

    def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
        return frameOpencvDnn, faceBoxes

    # Open RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        
        for faceBox in faceBoxes:
            # Extract face with padding
            face = frame[max(0, faceBox[1]-padding):
                        min(faceBox[3]+padding, frame.shape[0]-1),
                        max(0, faceBox[0]-padding):
                        min(faceBox[2]+padding, frame.shape[1]-1)]

            # Gender detection
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Age detection
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # Draw rectangle and label
            cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), 
                         (0, 255, 0), int(round(frame.shape[0]/150)), 8)
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', resultImg)
        frame = buffer.tobytes()

        # Yield the frame as byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

import cv2
import time
import triangulation as tri
import calibration
import numpy as np
import time
from ultralytics import YOLO
import math
from scipy.optimize import linear_sum_assignment

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # IoU
    return inter_area / union_area if union_area > 0 else 0

def create_cost_matrix(left_boxes, right_boxes, dy_threshold=10, iou_threshold=0.1):
    cost_matrix = []
    for lbox in left_boxes:
        lx, ly, lw, lh = lbox
        l_center = (lx + lw / 2, ly + lh / 2)

        costs = []
        for rbox in right_boxes:
            rx, ry, rw, rh = rbox
            r_center = (rx + rw / 2, ry + rh / 2)

            # Vertical proximity
            dy = abs(l_center[1] - r_center[1])
            if dy > dy_threshold:
                costs.append(float('inf'))  # Large cost for invalid matches
                continue

            # Compute IoU
            iou = compute_iou(lbox, rbox)
            if iou < iou_threshold:
                costs.append(float('inf'))  # Large cost for poor matches
                continue

            # Combine cost components
            cost = 1 - iou  # Lower cost for higher IoU
            costs.append(cost)

        cost_matrix.append(costs)

    return cost_matrix

def reorder_right_boxes(left_boxes, right_boxes):
    # Create cost matrix
    cost_matrix = create_cost_matrix(left_boxes, right_boxes)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reorder the right boxes based on the matching
    reordered_right_boxes = [right_boxes[c] for c in col_ind]
    
    return reordered_right_boxes

model=YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus",
              "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Stereo vision setup parameters  
B = 9               # Distance between the cameras [cm]
f = 8               # Camera lens's focal length [mm] from manufacturer's data
alpha = 56.6        # Camera field of view in the horizontal plane [degrees] from camera's datasheet


while True:
    success_right,frame_right=cap_right.read()
    success_left,frame_left=cap_left.read()

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    if not success_right or not success_left:                    
        break

    start = time.time()

    results_right=model.predict(frame_right)
    results_left=model.predict(frame_left)
    # print(results_right)
    # print(results_left)

    left_boxes = []
    right_boxes = []

    for r in results_right:
        boxes=r.boxes
        # print(boxes)
        for box in boxes:
            # print(box)
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            right_boxes.append((x1,y1,x2-x1,y2-y1))
            cv2.rectangle(frame_right,(x1,y1),(x2,y2),(0,255,0),5)
            # center_point_right = ((x1 + x2) / 2, (y1 + y2) / 2)
            conf=math.ceil(box.conf[0]*100)/100
            # print(conf)
            cls=box.cls[0]
            cls=int(cls)
            cv2.putText(frame_right,f'{classNames[cls]} {conf}',(x1,y1-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)

    for r in results_left:
        boxes=r.boxes
        # print(boxes)
        for box in boxes:
            # print(box)
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            left_boxes.append((x1,y1,x2-x1,y2-y1))
            cv2.rectangle(frame_left,(x1,y1),(x2,y2),(0,255,0),5)
            # center_point_left = ((x1 + x2) / 2, (y1 + y2) / 2)
            conf=math.ceil(box.conf[0]*100)/100
            # print(conf)
            cls=box.cls[0]
            cls=int(cls)
            cv2.putText(frame_left,f'{classNames[cls]} {conf}',(x1,y1-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
    
    right_detections = reorder_right_boxes(left_boxes, right_boxes)
    left_detections = left_boxes

    center_points_right = []
    center_points_left = []

    for box in right_detections:      
        x, y, w, h = box
        center_x = x + w / 2
        center_y = y + h / 2
        center_points_right.append((center_x, center_y))

    for box in left_detections:      
        x, y, w, h = box
        center_x = x + w / 2
        center_y = y + h / 2
        center_points_left.append((center_x, center_y))

    for i, center_point_right in center_points_right:
        center_point_left = center_points_left[i]
        depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)
        depth = depth * 205.8 # Multiply computer value with 205.8 to get real-life depth in [cm]. This factor was found manually.
        cv2.putText(frame_right, f"{depth:.2f} cm", (center_point_right), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        cv2.putText(frame_left, f"{depth:.2f} cm", (center_point_right), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        print("Depth: ", f"{depth:.2f} cm")
      
    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   

    cv2.imshow("frame right", frame_right) 
    cv2.imshow("frame left", frame_left)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
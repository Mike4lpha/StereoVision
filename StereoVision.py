import cv2
import time
import triangulation as tri
import calibration
import mediapipe as mp
import time

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)                    
cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Stereo vision setup parameters    
B = 9               # Distance between the cameras [cm]
f = 8               # Camera lens's focal length [mm] from manufacturer's data
alpha = 56.6        # Camera field of view in the horizontal plane [degrees] from camera's datasheet


with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while(cap_right.isOpened() and cap_left.isOpened()):

        success_right, frame_right = cap_right.read()
        success_left, frame_left = cap_left.read()

        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

        if not success_right or not success_left:                    
            break

        else:

            start = time.time()
    
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

            results_right = face_detection.process(frame_right)
            results_left = face_detection.process(frame_left)

            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

            ### CALCULATING DEPTH 

            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_right.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_left.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


            # If no face can be caught in one camera show text "TRACKING LOST"
            if not results_right.detections or not results_left.detections:
                cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            else:
                depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)
                depth = depth * 205.8 # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
                cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                print("Depth: ", str(round(depth,1)))

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
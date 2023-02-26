import cv2
import mediapipe as mp
import math
import gradio
import websocket
import threading
from flask import Flask, render_template, Response

global cap
cap = cv2.VideoCapture(0)

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

joint_indices = [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                 mp_pose.PoseLandmark.LEFT_ELBOW.value,
                 mp_pose.PoseLandmark.LEFT_HIP.value,
                 mp_pose.PoseLandmark.LEFT_KNEE.value,
                 mp_pose.PoseLandmark.RIGHT_HIP.value,
                 mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                 mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                 mp_pose.PoseLandmark.RIGHT_KNEE.value
                 ]

pushup_counter = 0
squat_counter = 0
is_plank = False
is_squat = False


class points:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calculate_angle(a, b, c):
    ab = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    bc = math.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)
    ac = math.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)
    cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
    angle_radians = math.acos(cos_angle)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def gen_frames():
    global cap
    global pushup_counter
    global squat_counter
    global is_pushup
    global is_squat
    cap = cv2.VideoCapture(0)
    while True:
    
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)
        annotated_frame = frame.copy()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            joint_positions = []
            for index in joint_indices:
                landmark = landmarks[index]
                joint_positions.append((landmark.x, landmark.y))

            angles = []
            for i in range(len(joint_positions) - 2):
                angle = calculate_angle(*joint_positions[i:i+3])
                angles.append(angle)

            for i, angle in enumerate(angles):
                cv2.putText(annotated_frame, f'{i}: {angle:.2f}', (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
            left_shoulder, left_elbow, left_hip, left_knee, right_hip, right_elbow, right_shoulder, right_knee = joint_positions

            
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_hip)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_hip)

            left_hip_angle = calculate_angle(left_shoulder, left_hip, right_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            if left_elbow_angle < 100 and right_elbow_angle < 100 :
                if not is_pushup:
                    is_plank = True
                    pushup_counter += 1
            else:
                is_pushup = False
            
            if left_hip_angle <152.5  and right_hip_angle <152.5 :
                if not is_squat:
                    is_squat = True
                    squat_counter += 1
            else:
                is_squat = False
        if is_squat:
            cv2.putText(annotated_frame, f'Bad form', (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_frame, f'Good form', (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
       
        # Display the resulting frame
        frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # cv2.imshow('Pose Detection', annotated_frame)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break
        # ret, buffer = cv2.imencode('.jpg', frame)
        # frame = buffer.tobytes()
        # yield (b'--frame\r\n'
        #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

cap.release()
cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # defining server ip address and port
    app.run( debug=True)

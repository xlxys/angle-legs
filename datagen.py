import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Process leg angles from video.')
parser.add_argument('--video', type=str, required=True, help='Path to video file')
parser.add_argument('--display', action='store_true', help='Display video output')
args = parser.parse_args()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
mp_draw = mp.solutions.drawing_utils

# Load video file
cap = cv2.VideoCapture(args.video)

# Get total number of frames for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Data collection list
data = []
frame_count = 0

for _ in tqdm(range(total_frames), desc="Processing video"):
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        def get_coords(index):
            lm = landmarks[index]
            return int(lm.x * w), int(lm.y * h)

        try:
            # Right leg
            r_hip = get_coords(24)
            r_knee = get_coords(26)
            r_ankle = get_coords(28)
            r_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # Left leg
            l_hip = get_coords(23)
            l_knee = get_coords(25)
            l_ankle = get_coords(27)
            l_angle = calculate_angle(l_hip, l_knee, l_ankle)

            data.append({
                'frame': frame_count,
                'r_hip_x': r_hip[0], 'r_hip_y': r_hip[1],
                'r_knee_x': r_knee[0], 'r_knee_y': r_knee[1],
                'r_ankle_x': r_ankle[0], 'r_ankle_y': r_ankle[1],
                'r_angle': r_angle,
                'l_hip_x': l_hip[0], 'l_hip_y': l_hip[1],
                'l_knee_x': l_knee[0], 'l_knee_y': l_knee[1],
                'l_ankle_x': l_ankle[0], 'l_ankle_y': l_ankle[1],
                'l_angle': l_angle
            })

            if args.display:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f'R: {int(r_angle)}', (r_knee[0]-40, r_knee[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f'L: {int(l_angle)}', (l_knee[0]-40, l_knee[1]-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                resized = cv2.resize(frame, (480*2, 360*2))
                cv2.imshow('Pose Detection', resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except:
            continue

cap.release()
cv2.destroyAllWindows()

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv('leg_angles.csv', index=False)
print("Saved angles to leg_angles.csv")
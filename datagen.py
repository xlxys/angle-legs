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

    if results.pose_world_landmarks:
        landmarks = results.pose_world_landmarks.landmark

        def get_coords(index):
            lm = landmarks[index]
            return lm.x, lm.y, lm.z

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
                'r_hip_y': r_hip[0], 'r_hip_x': r_hip[1], 'r_hip_z': r_hip[2],
                'r_knee_y': r_knee[0], 'r_knee_x': r_knee[1], 'r_knee_z': r_knee[2],
                'r_ankle_y': r_ankle[0], 'r_ankle_x': r_ankle[1], 'r_ankle_z': r_ankle[2],
                'r_angle': r_angle,
                'l_hip_y': l_hip[0], 'l_hip_x': l_hip[1], 'l_hip_z': l_hip[2],
                'l_knee_y': l_knee[0], 'l_knee_x': l_knee[1], 'l_knee_z': l_knee[2],
                'l_ankle_y': l_ankle[0], 'l_ankle_x': l_ankle[1], 'l_ankle_z': l_ankle[2],
                'l_angle': l_angle
            })

            if args.display and results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                h, w, _ = frame.shape

                # Use the midpoint between left and right hip as reference point
                ref_x = int((results.pose_landmarks.landmark[23].x + results.pose_landmarks.landmark[24].x) / 2 * w)
                ref_y = int((results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2 * h)
                cv2.drawMarker(frame, (ref_x, ref_y), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(frame, 'Hip Center (0,0,0)', (ref_x + 5, ref_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Display world coordinates for key landmarks
                for name, idx in [('R_Knee', 26), ('R_Hip', 24), ('R_Ankle', 28),
                                  ('L_Knee', 25), ('L_Hip', 23), ('L_Ankle', 27)]:
                    lm = landmarks[idx]
                    px, py = int(results.pose_landmarks.landmark[idx].x * w), int(results.pose_landmarks.landmark[idx].y * h)
                    coords = f"{name}: ({lm.x:.2f}, {lm.y:.2f}, {lm.z:.2f})"
                    cv2.putText(frame, coords, (px + 5, py), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)

                resized = cv2.resize(frame, (480, 360))
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

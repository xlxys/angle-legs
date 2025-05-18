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
parser.add_argument('--output', type=str, default='leg_angles_v2.csv', help='Output CSV file name')
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

# Function to calculate distance between two 3D points
def distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to calculate distance between two 2D points
def distance_2d(p1, p2):
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

# Function to calculate inclination angle from vertical
def angle_from_vertical(v):
    vertical = np.array([0, 1, 0])
    v = np.array(v)
    cosine = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

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

    if results.pose_world_landmarks and results.pose_landmarks:
        world_landmarks = results.pose_world_landmarks.landmark
        image_landmarks = results.pose_landmarks.landmark

        def get_world_coords(index):
            lm = world_landmarks[index]
            return lm.x, lm.y, lm.z

        def get_image_coords(index):
            lm = image_landmarks[index]
            return lm.x, lm.y

        try:
            # World landmarks for angles/distances
            r_hip = get_world_coords(24)
            r_knee = get_world_coords(26)
            r_ankle = get_world_coords(28)
            r_heel = get_world_coords(30)
            r_shoulder = get_world_coords(12)

            l_hip = get_world_coords(23)
            l_knee = get_world_coords(25)
            l_ankle = get_world_coords(27)
            l_heel = get_world_coords(29)
            l_shoulder = get_world_coords(11)

            # Angles
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
            r_ankle_angle = calculate_angle(r_knee, r_ankle, r_heel)
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            l_ankle_angle = calculate_angle(l_knee, l_ankle, l_heel)

            # Lengths
            r_thigh_length = distance_3d(r_hip, r_knee)
            r_calf_length = distance_3d(r_knee, r_ankle)
            l_thigh_length = distance_3d(l_hip, l_knee)
            l_calf_length = distance_3d(l_knee, l_ankle)
            hip_width = distance_3d(r_hip, l_hip)

            # Flexion angles in sagittal plane
            r_thigh_vec = np.array(r_hip) - np.array(r_knee)
            r_shank_vec = np.array(r_ankle) - np.array(r_knee)
            l_thigh_vec = np.array(l_hip) - np.array(l_knee)
            l_shank_vec = np.array(l_ankle) - np.array(l_knee)

            r_thigh_angle_vert = angle_from_vertical(r_thigh_vec)
            r_shank_angle_vert = angle_from_vertical(r_shank_vec)
            l_thigh_angle_vert = angle_from_vertical(l_thigh_vec)
            l_shank_angle_vert = angle_from_vertical(l_shank_vec)

            r_knee_flexion = abs(r_thigh_angle_vert - r_shank_angle_vert)
            l_knee_flexion = abs(l_thigh_angle_vert - l_shank_angle_vert)


            data.append({
                'frame': frame_count,
                'r_hip_x': r_hip[0], 'r_hip_y': r_hip[1], 'r_hip_z': r_hip[2],
                'r_knee_x': r_knee[0], 'r_knee_y': r_knee[1], 'r_knee_z': r_knee[2],
                'r_ankle_x': r_ankle[0], 'r_ankle_y': r_ankle[1], 'r_ankle_z': r_ankle[2],
                'r_angle': r_knee_angle,
                'l_hip_x': l_hip[0], 'l_hip_y': l_hip[1], 'l_hip_z': l_hip[2],
                'l_knee_x': l_knee[0], 'l_knee_y': l_knee[1], 'l_knee_z': l_knee[2],
                'l_ankle_x': l_ankle[0], 'l_ankle_y': l_ankle[1], 'l_ankle_z': l_ankle[2],
                'l_angle': l_knee_angle,
                'r_knee_angle': r_knee_angle,
                'r_hip_angle': r_hip_angle,
                'r_ankle_angle': r_ankle_angle,
                'l_knee_angle': l_knee_angle,
                'l_hip_angle': l_hip_angle,
                'l_ankle_angle': l_ankle_angle,
                'r_thigh_length': r_thigh_length,
                'r_calf_length': r_calf_length,
                'l_thigh_length': l_thigh_length,
                'l_calf_length': l_calf_length,
                'hip_width': hip_width,
                'r_knee_flexion': r_knee_flexion,
                'l_knee_flexion': l_knee_flexion,
            })

            if args.display:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Prepare text for display
                display_text = [
                    f"Frame: {frame_count}",
                    f"R Knee Angle: {r_knee_angle:.1f}°",
                    f"R Hip Angle: {r_hip_angle:.1f}°",
                    f"R Ankle Angle: {r_ankle_angle:.1f}°",
                    f"L Knee Angle: {l_knee_angle:.1f}°",
                    f"L Hip Angle: {l_hip_angle:.1f}°",
                    f"L Ankle Angle: {l_ankle_angle:.1f}°",
                    f"R Thigh Length: {r_thigh_length:.2f} m",
                    f"R Calf Length: {r_calf_length:.2f} m",
                    f"L Thigh Length: {l_thigh_length:.2f} m",
                    f"L Calf Length: {l_calf_length:.2f} m",
                    f"Hip Width: {hip_width:.2f} m"
                ]

                # Draw the text
                for i, line in enumerate(display_text):
                    cv2.putText(frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                resized = cv2.resize(frame, (960, 720))  # Resize for better visibility
                cv2.imshow('Pose Detection', resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue

cap.release()
cv2.destroyAllWindows()

# Create DataFrame
df = pd.DataFrame(data)
# Save to CSV
df.to_csv(args.output, index=False)
print("Saved angles and widths to leg_angles.csv")

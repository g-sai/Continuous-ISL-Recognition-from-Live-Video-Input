import cv2
import mediapipe as mp
import numpy as np
import os


"""
This script processes an input image to detect hand and upper body landmarks using MediaPipe. 
It creates two output masks: one showing only key points and another with key points connected by lines.
"""

output_folder = "/path/to/output_images_dir"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_image_path = "/path/to/frame.png"
image = cv2.imread(input_image_path)

cv2.imwrite(os.path.join(output_folder, "input_frame.jpg"), image)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(output_folder, "rgb_frame.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

mask_points = np.zeros_like(image)
mask_joined = np.zeros_like(image)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    results_hands = hands.process(image_rgb)

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results_pose = pose.process(image_rgb)

h, w, _ = image.shape

if results_hands.multi_hand_landmarks:
    for hand_landmarks in results_hands.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(mask_points, (cx, cy), 3, (255, 255, 255), -1)
            cv2.circle(mask_joined, (cx, cy), 3, (255, 255, 255), -1)
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(mask_joined, start_point, end_point, (255, 255, 255), 2)

upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24]
pose_points = {}

if results_pose.pose_landmarks:
    for idx in upper_body_indices:
        landmark = results_pose.pose_landmarks.landmark[idx]
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        pose_points[idx] = (cx, cy)
        cv2.circle(mask_points, (cx, cy), 3, (255, 255, 255), -1)
        cv2.circle(mask_joined, (cx, cy), 3, (255, 255, 255), -1)

    if 11 in pose_points and 13 in pose_points:
        cv2.line(mask_joined, pose_points[11], pose_points[13], (255, 255, 255), 2)
    if 13 in pose_points and 15 in pose_points:
        cv2.line(mask_joined, pose_points[13], pose_points[15], (255, 255, 255), 2)
    if 12 in pose_points and 14 in pose_points:
        cv2.line(mask_joined, pose_points[12], pose_points[14], (255, 255, 255), 2)
    if 14 in pose_points and 16 in pose_points:
        cv2.line(mask_joined, pose_points[14], pose_points[16], (255, 255, 255), 2)
    if 11 in pose_points and 12 in pose_points:
        cv2.line(mask_joined, pose_points[11], pose_points[12], (255, 255, 255), 2)
    if 11 in pose_points and 23 in pose_points:
        cv2.line(mask_joined, pose_points[11], pose_points[23], (255, 255, 255), 2)
    if 12 in pose_points and 24 in pose_points:
        cv2.line(mask_joined, pose_points[12], pose_points[24], (255, 255, 255), 2)
    if 23 in pose_points and 24 in pose_points:
        cv2.line(mask_joined, pose_points[23], pose_points[24], (255, 255, 255), 2)

cv2.imwrite(os.path.join(output_folder, "coordinates_mask_points.jpg"), mask_points)
cv2.imwrite(os.path.join(output_folder, "coordinates_mask_joined.jpg"), mask_joined)

print("Images saved in folder:", output_folder)

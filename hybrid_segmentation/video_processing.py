import numpy as np
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List
import numpy as np
import math


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_frame_for_hands(frame):

    """
    Processes a single video frame to detect hand landmarks using MediaPipe.

    Args:
        frame (np.ndarray): A single frame from a video in BGR format.

    Returns:
        np.ndarray: A 1D array of size 126 containing hand landmark coordinates.
                    
    """

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
        results = hands.process(image_rgb)
    
    frame_features = np.zeros(126)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2:
                break
            for j, landmark in enumerate(hand_landmarks.landmark):
                idx = i * 63 + j * 3
                frame_features[idx:idx+3] = [landmark.x, landmark.y, landmark.z]
    return frame_features


def extract_frames(video_path: str) -> Tuple[List[np.ndarray], int]:

    """
    Extracts all frames from a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        Tuple[List[np.ndarray], int]: A list of frames (NumPy array) and the total frame count.
    """

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    
    cap.release()
    return frames, len(frames)


def process_frame_batch(frames: List[np.ndarray]) -> List[np.ndarray]:
    return [process_frame_for_hands(frame) for frame in frames]


def process_video(video_path: str, num_threads: int = None) -> np.ndarray:

    """
    Processes a video to extract hand landmarks using multi-threading.

    Args:
        video_path (str): The path to the video file.
        num_threads (int, optional): The number of threads to use. Defaults to the minimum of 8 or the number of CPU cores.

    Returns:
        np.ndarray: A NumPy array of shape (num_frames, 126) containing hand landmark features for each frame.
    """

    frames, total_frames = extract_frames(video_path)
    
    if not frames:
        return np.array([])
    
    if num_threads is None:
        num_threads = min(8, cv2.getNumberOfCPUs()) 

    batch_size = math.ceil(len(frames) / num_threads)
    
    frame_batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_frame_batch, frame_batches))
    
    frame_features_list = []
    for batch_result in results:
        frame_features_list.extend(batch_result)
    
    return np.array(frame_features_list)

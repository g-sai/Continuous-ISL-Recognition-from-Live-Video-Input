import os
import cv2

def get_video_frame_count(video_path):
    
    """
    Get the total number of frames in a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        int: The number of frames in the video. Returns 0 if the video cannot be opened.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0  
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def find_max_frames_in_directory(directory_path):
    
    """
    Traverse a directory and its subdirectories to find the video with the maximum number of frames.

    Args:
        directory_path (str): The path to the directory containing video files.

    Returns:
        tuple: A tuple containing:
            - max_frames (int): The highest frame count found.
            - max_frame_video (str): The file path of the video with the highest frame count. 
    """

    max_frames = 0
    max_frame_video = None

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv')):
                video_path = os.path.join(root, file)
                frame_count = get_video_frame_count(video_path)
                if frame_count > max_frames:
                    max_frames = frame_count
                    max_frame_video = video_path

    return max_frames, max_frame_video

directory_path = '/path/to/dataset_dir'

max_frames, max_frame_video = find_max_frames_in_directory(directory_path)

if max_frame_video:
    print(f"Video with max frames: {max_frame_video}")
    print(f"Maximum number of frames: {max_frames}")
else:
    print("No video files found in the directory.")

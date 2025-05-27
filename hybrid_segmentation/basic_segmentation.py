import numpy as np


def predict_segments_with_padding(video_features, boundaries):

    """
    Ensures segment boundaries cover the entire length of the video and returns 
    adjusted segment tuples.
    
    Args:
        video_features (np.ndarray): The extracted video feature representations.
        boundaries (list): List of segment boundary indices.

    Returns:
        list: List of tuples where each tuple (start, end) represents a video segment.
    """

    predictions = []
    
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    
    if boundaries[-1] != len(video_features):
        boundaries.append(len(video_features))
    
    l=[]
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        l.append((start,end))
        
    return l

def detect_segments(video_features, y_min=0.8, y_max=1.0, consecutive_frames=3, min_gap=30):
    
    """
    Detects segments in a video based on hand position thresholds.

    Args:
        video_features (np.ndarray): The extracted video feature representations.
        y_min (float): Minimum Y-coordinate threshold for detecting hand positions.
        y_max (float): Maximum Y-coordinate threshold for detecting hand positions.
        consecutive_frames (int): Minimum number of consecutive frames to qualify as a segment.
        min_gap (int): Minimum frame gap between segments to avoid merging.

    Returns:
        list: A list of tuples, where each tuple (start, end) represents detected segments.
    """

    left_hand_y = video_features[:, 1]
    right_hand_y = video_features[:, 64]
    
    in_range = np.logical_and(
        np.logical_and(left_hand_y >= y_min, left_hand_y <= y_max),
        np.logical_and(right_hand_y >= y_min, right_hand_y <= y_max)
    )
    
    segments = []
    count = 0
    start = None
    last_end = -min_gap  
    
    for i, in_range_frame in enumerate(in_range):
        if in_range_frame:
            if start is None:
                start = i
            count += 1
        else:
            if count >= consecutive_frames and start is not None and (start - last_end) > min_gap:
                segments.append((start, i))
                last_end = i 
            count = 0
            start = None
    
    if count >= consecutive_frames and start is not None and (start - last_end) > min_gap:
        segments.append((start, len(in_range)))
    
    return segments


def robust_segmentation(video_features, y_min=0.8, y_max=1.0, consecutive_frames=3, min_gap=30):

    """
    Performs robust segmentation of video features based on detected hand positions.

    Args:
        video_features (np.ndarray): The extracted video feature representations.
        y_min (float): Minimum Y-coordinate threshold for hand detection.
        y_max (float): Maximum Y-coordinate threshold for hand detection.
        consecutive_frames (int): Minimum number of consecutive frames to form a valid segment.
        min_gap (int): Minimum frame gap between two segments.

    Returns:
        tuple: 
            - List of segmented video feature arrays.
            - Placeholder (None) for future additional return values.
            - List of segment boundary indices.
    """

    segments = detect_segments(video_features, y_min, y_max, consecutive_frames, min_gap)
    video_segments = [video_features[start:end] for start, end in segments]
    boundaries = [end for _, end in segments]
    
    return video_segments, None, boundaries
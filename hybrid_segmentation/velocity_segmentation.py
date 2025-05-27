import os
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import numpy as np



def velocity_jerk_segmentation(video_features, desired_segments, threshold_multiplier=0.02, min_segment_length=10, min_pause_length=5, output_dir='/path/to/graphs_dir'):

    """
    Segments a sign language video sequence using velocity, acceleration, jerk, and jerk rate 
    computed from hand motion features.

    Parameters:
    - video_features (numpy.ndarray): A 2D array where each row represents a frame, 
      and columns correspond to hand motion features.
    - desired_segments (int): The target number of segments for the video.
    - threshold_multiplier (float): A multiplier to dynamically adjust the threshold 
      for detecting pauses.
    - min_segment_length (int): Minimum allowed segment length.
    - min_pause_length (int): Minimum pause duration to be considered a segment boundary.
    - output_dir (str): Directory to save segmentation analysis plots.

    Returns:
    - segments (list of tuples): List of (start_frame, end_frame) tuples representing 
      segmented video parts.
    """


    os.makedirs(output_dir, exist_ok=True)
    
    left_hand_y = video_features[:, 1]
    right_hand_y = video_features[:, 64]
    frames = np.arange(len(video_features))

    def calculate_derivatives(signal, window_size=5):
        velocity = np.gradient(signal)
        acceleration = np.gradient(velocity)
        jerk = np.gradient(acceleration)
        jerk_rate = np.gradient(jerk)
        
        velocity = medfilt(velocity, kernel_size=window_size)
        acceleration = medfilt(acceleration, kernel_size=window_size)
        jerk = medfilt(jerk, kernel_size=window_size)
        jerk_rate = medfilt(jerk_rate, kernel_size=window_size)
        
        return velocity, acceleration, jerk, jerk_rate

    left_v, left_a, left_j, left_jr = calculate_derivatives(left_hand_y)
    right_v, right_a, right_j, right_jr = calculate_derivatives(right_hand_y)

    velocity_magnitude = np.sqrt(left_v**2 + right_v**2)
    acceleration_magnitude = np.sqrt(left_a**2 + right_a**2)
    jerk_magnitude = np.sqrt(left_j**2 + right_j**2)
    jerk_rate_magnitude = np.sqrt(left_jr**2 + right_jr**2)

    def normalize_signal(signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

    norm_velocity = normalize_signal(velocity_magnitude)
    norm_acceleration = normalize_signal(acceleration_magnitude)
    norm_jerk = normalize_signal(jerk_magnitude)
    norm_jerk_rate = normalize_signal(jerk_rate_magnitude)

    movement_metric = (0.5 * norm_velocity + 
                      0.3 * norm_acceleration + 
                      0.15 * norm_jerk + 
                      0.05 * norm_jerk_rate)

    def find_pause_regions(movement_metric, base_threshold=0.02, window_size=5):
        local_mean = np.convolve(movement_metric, 
                                np.ones(window_size)/window_size, 
                                mode='same')
        local_std = np.std(movement_metric)
        
        dynamic_threshold = base_threshold * (1 + local_mean) * (1 + local_std)
        
        pauses = movement_metric < dynamic_threshold
        
        pause_regions = []
        pause_start = None
        
        for i in range(len(pauses)):
            if pauses[i] and pause_start is None:
                pause_start = i
            elif not pauses[i] and pause_start is not None:
                pause_end = i
                pause_length = pause_end - pause_start
                if pause_length >= min_pause_length:
                    center_point = pause_start + pause_length // 2
                    pause_regions.append({
                        'start': pause_start,
                        'end': pause_end,
                        'center': center_point,
                        'length': pause_length
                    })
                pause_start = None
        
        if pause_start is not None:
            pause_length = len(pauses) - pause_start
            if pause_length >= min_pause_length:
                center_point = pause_start + pause_length // 2
                pause_regions.append({
                    'start': pause_start,
                    'end': len(pauses),
                    'center': center_point,
                    'length': pause_length
                })
        
        return pause_regions, pauses

    pause_regions, pauses = find_pause_regions(movement_metric, threshold_multiplier)
    
    segments = []
    if pause_regions:
        if pause_regions[0]['start'] > 0:
            segments.append((0, pause_regions[0]['center']))
        
        for i in range(len(pause_regions)-1):
            segments.append((pause_regions[i]['center'], pause_regions[i+1]['center']))
        
        if pause_regions[-1]['end'] < len(video_features):
            segments.append((pause_regions[-1]['center'], len(video_features)))
    else:
        segments = [(0, len(video_features))]
    
    while len(segments) < desired_segments:
        longest_idx = max(range(len(segments)), 
                         key=lambda i: segments[i][1] - segments[i][0])
        start, end = segments[longest_idx]
        
        segment_movement = movement_metric[start:end]
        split_point = start + np.argmin(segment_movement)
        
        segments[longest_idx] = (start, split_point)
        segments.insert(longest_idx + 1, (split_point, end))
    
    while len(segments) > desired_segments:
        min_combined_length = float('inf')
        merge_idx = 0
        
        for i in range(len(segments)-1):
            combined_length = segments[i+1][1] - segments[i][0]
            if combined_length < min_combined_length:
                min_combined_length = combined_length
                merge_idx = i
        
        segments[merge_idx] = (segments[merge_idx][0], segments[merge_idx+1][1])
        segments.pop(merge_idx + 1)

    plt.figure(figsize=(15, 12))
    
    plt.subplot(411)
    plt.plot(frames, movement_metric, label='Movement Metric')
    plt.plot(frames, [threshold_multiplier] * len(frames), 'r--', label='Base Threshold')
    plt.fill_between(frames, 0, 1, where=pauses, alpha=0.2, color='red', label='Detected Pauses')
    
    for pause in pause_regions:
        plt.axvline(x=pause['center'], color='purple', alpha=0.5, linestyle=':', 
                   label='Pause Center' if pause == pause_regions[0] else '')
    
    plt.title('Movement Metric and Detected Pauses')
    plt.legend()
    
    plt.subplot(412)
    plt.plot(frames, norm_velocity, label='Normalized Velocity')
    plt.title('Normalized Velocity')
    plt.legend()
    
    plt.subplot(413)
    plt.plot(frames, norm_jerk, label='Normalized Jerk')
    plt.title('Normalized Jerk')
    plt.legend()
    
    plt.subplot(414)
    plt.plot(frames, norm_jerk_rate, label='Normalized Jerk Rate')
    plt.title('Normalized Jerk Rate')
    plt.legend()
    
    for ax in plt.gcf().axes:
        for start, end in segments:
            ax.axvline(x=start, color='g', alpha=0.5, linestyle='--')
            ax.axvline(x=end, color='r', alpha=0.5, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_velocity_segmentation_analysis.png'))
    plt.close()

    return segments
import numpy as np
from scipy.signal import find_peaks, medfilt
import pywt
import matplotlib.pyplot as plt



def dwt_segmentation(video_features, num_segments, min_segment_length=30, wavelet='db4', level=3):

    """
    Performs segmentation of video features using Discrete Wavelet Transform (DWT).
    

    Args:
        video_features (np.ndarray): A 2D numpy array where each row represents a frame 
                                     and each column represents a feature.
        num_segments (int): The desired number of segments.
        min_segment_length (int, optional): Minimum allowed length of each segment. 
                                            Defaults to 30 frames.
        wavelet (str, optional): Wavelet type used for DWT. Defaults to 'db4'.
        level (int, optional): Decomposition level for DWT. Defaults to 3.

    Returns:
        list: A list of tuples where each tuple (start, end) represents the frame indices 
              of a detected segment.
    """

    left_hand_features = video_features[:, :63]
    right_hand_features = video_features[:, 63:]
    
    
    def compute_motion_signal(features):

        """
        Computes motion signals by summing position, velocity, and acceleration.

        Args:
            features (np.ndarray): A 2D array of hand feature coordinates.

        Returns:
            np.ndarray: A 1D array representing the motion signal over time.
        """

        motion = np.zeros(len(features))
        for i in range(0, features.shape[1], 3):
            pos = features[:, i:i+3]
            vel = np.diff(pos, axis=0, prepend=pos[0].reshape(1, -1))
            acc = np.diff(vel, axis=0, prepend=vel[0].reshape(1, -1))
            motion += np.sum(pos**2, axis=1) + np.sum(vel**2, axis=1) + np.sum(acc**2, axis=1)
        return motion
    
    left_motion = compute_motion_signal(left_hand_features)
    right_motion = compute_motion_signal(right_hand_features)
    
    
    def apply_dwt_and_get_energy(signal):

        """
        Applies Discrete Wavelet Transform (DWT) to a motion signal and extracts energy.

        Args:
            signal (np.ndarray): The input motion signal.
        
        Returns:
            np.ndarray: A transformed energy representation of the signal.
        """

        coeffs = pywt.wavedec(signal, wavelet, level=level)
        details = [np.abs(coeff) for coeff in coeffs[1:]]
        weights = [2**i for i in range(len(details))]
        weighted_details = [w * d for w, d in zip(weights, details)]
        reconstructed = []
        for detail in weighted_details:
            reconstructed.extend(np.repeat(detail, 2**(level-len(reconstructed))))
        return np.array(reconstructed[:len(signal)])
    
    left_energy = apply_dwt_and_get_energy(left_motion)
    right_energy = apply_dwt_and_get_energy(right_motion)
    
    combined_energy = (left_energy / np.max(left_energy) + 
                      right_energy / np.max(right_energy))
    
    window_size = min_segment_length // 2
    smoothed_energy = medfilt(combined_energy, kernel_size=window_size)
    

    def find_optimal_transitions(energy, n_segments, total_frames):
        
        """
        Identifies optimal transition points for segmentation using energy gradients.

        Args:
            energy (np.ndarray): The smoothed energy signal.
            n_segments (int): Number of desired segments.
            total_frames (int): Total number of frames in the video.

        Returns:
            list: A sorted list of frame indices representing segment boundaries.
        """

        if n_segments <= 1:
            return []
        gradient = np.gradient(energy)
        gradient_magnitude = np.abs(gradient)
        
        smooth_gradient = medfilt(gradient_magnitude, kernel_size=window_size)
        potential_points = []
        min_distance = min_segment_length
        peaks, _ = find_peaks(smooth_gradient, distance=min_distance)
        peak_scores = []
        for peak in peaks:
            gradient_score = smooth_gradient[peak]
            energy_change = np.abs(energy[min(peak+5, len(energy)-1)] - 
                                 energy[max(peak-5, 0)])
            score = gradient_score * energy_change
            peak_scores.append((peak, score))
        sorted_peaks = sorted(peak_scores, key=lambda x: x[1], reverse=True)
        selected_points = []
        for peak, _ in sorted_peaks:
            valid_point = True
            for existing_point in selected_points:
                if abs(peak - existing_point) < min_distance:
                    valid_point = False
                    break
            
            if valid_point:
                selected_points.append(peak)
                if len(selected_points) == n_segments - 1:
                    break
        
        print("\n",selected_points,"\n")
        while len(selected_points) < n_segments - 1:
            ideal_spacing = total_frames // n_segments
            potential_point = ideal_spacing * (len(selected_points) + 1)
            if potential_point < total_frames - min_distance:
                selected_points.append(potential_point)
        
        return sorted(selected_points)
    
    total_frames = len(video_features)
    transition_points = find_optimal_transitions(smoothed_energy, num_segments, total_frames)
    
    segments = []
    if not transition_points:
        segments = [(0, total_frames)]
    else:
        segments = [(0, transition_points[0])]
        segments.extend([(transition_points[i], transition_points[i+1]) 
                        for i in range(len(transition_points)-1)])
        segments.append((transition_points[-1], total_frames))
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(left_motion, label='Left Hand Motion', alpha=0.7)
    plt.plot(right_motion, label='Right Hand Motion', alpha=0.7)
    plt.title('Hand Motion Signals')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(combined_energy, label='Combined Energy', alpha=0.5)
    plt.plot(smoothed_energy, label='Smoothed Energy', linewidth=2)
    plt.title(f'Energy Signal (Targeting {num_segments} segments)')
    for point in transition_points:
        plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(smoothed_energy, label='Energy', color='b', alpha=0.7)
    for start, end in segments:
        plt.axvspan(start, end, alpha=0.2, color='red')
    plt.title(f'Final Segmentation ({len(segments)} segments)')
    
    plt.tight_layout()
    plt.savefig('/path/to/dwt_final_segmentation.png')
    plt.close()
    
    assert len(segments) == num_segments, f"Expected {num_segments} segments, but got {len(segments)}"
    return segments

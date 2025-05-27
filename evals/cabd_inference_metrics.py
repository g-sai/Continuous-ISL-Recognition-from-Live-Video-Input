import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import keras
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter



class SignLanguageProcessor:
    
    def __init__(self, model_path, encoder_path):

        """Initialize the processor with model and encoder paths"""

        self.model, self.label_encoder = self.load_model_and_encoder(model_path, encoder_path)
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,  
            model_complexity=1,  
            enable_segmentation=False,  
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
            refine_face_landmarks=False  
        )
        
        self.upper_body_points = [11, 12, 13, 14, 15, 16, 23, 24]
        
        self.upper_body_landmarks = np.zeros(24)
        self.left_hand_landmarks = np.zeros(63)
        self.right_hand_landmarks = np.zeros(63)

    def load_model_and_encoder(self, model_path, encoder_path):
        try:
            keras.config.enable_unsafe_deserialization()
            model = load_model(model_path)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            return model, label_encoder
        except Exception as e:
            print(f"Error loading model or encoder: {str(e)}")
            return None, None

    def process_frame_for_upper_body(self, frame, results):

        """Process a single frame to extract upper body and hand landmarks"""

        try:
            self.upper_body_landmarks.fill(0)
            self.left_hand_landmarks.fill(0)
            self.right_hand_landmarks.fill(0)
            
            if results.pose_landmarks:
                for i, idx in enumerate(self.upper_body_points):
                    landmark = results.pose_landmarks.landmark[idx]
                    self.upper_body_landmarks[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]
            
            if results.left_hand_landmarks:
                for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                    self.left_hand_landmarks[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]
            
            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    self.right_hand_landmarks[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]
            
            return np.concatenate([
                self.upper_body_landmarks,
                self.left_hand_landmarks,
                self.right_hand_landmarks
            ])
        
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return np.zeros(150)

    def predict_segment(self, features, confidence_threshold=0.1):

        """Make predictions for a segment with confidence thresholding"""

        try:
            features_array = np.array(features)
            
            if len(features_array.shape) == 2:
                features_array = np.expand_dims(features_array, axis=0)
            
            predictions = self.model.predict(features_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]
            
            if confidence >= confidence_threshold:
                predicted_class = self.label_encoder.inverse_transform([predicted_class_index])[0]
            else:
                predicted_class = "Uncertain"
            
            return {
                'top_prediction': predicted_class,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None

def get_hand_boxes(results, image_shape):

    """Extract hand bounding boxes from MediaPipe results using holistic model output."""

    boxes = [None, None] 
    
    if results.left_hand_landmarks:
        landmarks = np.array([[lm.x * image_shape[1], lm.y * image_shape[0]] 
                            for lm in results.left_hand_landmarks.landmark])
        padding = 20
        min_coords = landmarks.min(axis=0)
        max_coords = landmarks.max(axis=0)
        
        boxes[0] = (
            max(0, int(min_coords[0]) - padding),
            max(0, int(min_coords[1]) - padding),
            min(image_shape[1], int(max_coords[0]) + padding),
            min(image_shape[0], int(max_coords[1]) + padding)
        )
    
    if results.right_hand_landmarks:
        landmarks = np.array([[lm.x * image_shape[1], lm.y * image_shape[0]] 
                            for lm in results.right_hand_landmarks.landmark])
        padding = 20
        min_coords = landmarks.min(axis=0)
        max_coords = landmarks.max(axis=0)
        
        boxes[1] = (
            max(0, int(min_coords[0]) - padding),
            max(0, int(min_coords[1]) - padding),
            min(image_shape[1], int(max_coords[0]) + padding),
            min(image_shape[0], int(max_coords[1]) + padding)
        )
    
    return boxes

def calculate_box_metrics(box1, box2):

    """Calculate IoU and distance between boxes efficiently."""

    if not all(box1) or not all(box2):
        return 0, float('inf')
    
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    if xB <= xA or yB <= yA:
        return 0, distance
    
    interArea = (xB - xA) * (yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    if box1Area <= 0 or box2Area <= 0:
        return 0, distance
    
    iou = interArea / float(box1Area + box2Area - interArea)
    return max(0, min(1, iou)), distance



def segment_video(video_path, processor, iou_threshold=0.85, pause_duration_threshold=8, proximity_threshold_factor=0.2):
   
    """Optimized video segmentation function without landmark visualization."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    prev_boxes = [None, None]
    pause_counters = [0, 0]
    state = "RUNNING"
    segments = []
    start_frame = None
    frame_count = 0
    pause_start_frame = None
    current_segment_features = []
    predicted_words = []
    segment_predictions = []
    current_segment_prediction = None

    l3=[]
    temp_eng_sentence=''
    
    proximity_threshold = proximity_threshold_factor * frame_width
    bottom_threshold = frame_height * 0.7
    
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = processor.holistic.process(image_rgb)
        
        frame_features = processor.process_frame_for_upper_body(frame, results)
        if frame_features is not None:
            current_segment_features.append(frame_features)
        
        current_boxes = get_hand_boxes(results, frame.shape)
        
        for i in range(2):
            if prev_boxes[i] is not None and current_boxes[i] is not None:
                iou, _ = calculate_box_metrics(current_boxes[i], prev_boxes[i])
                if iou > iou_threshold:
                    pause_counters[i] += 1
                else:
                    pause_counters[i] = 0
            else:
                pause_counters[i] = 0
        
        both_hands_paused = all(pc >= pause_duration_threshold for pc in pause_counters)
        hands_in_proximity = False
        hands_in_bottom = False
        
        if all(current_boxes):
            _, distance = calculate_box_metrics(current_boxes[0], current_boxes[1])
            hands_in_proximity = distance < proximity_threshold
            hands_in_bottom = (current_boxes[0][3] > bottom_threshold and 
                             current_boxes[1][3] > bottom_threshold)
        
        if both_hands_paused and hands_in_proximity and hands_in_bottom:
            if state == "RUNNING":
                if start_frame is not None and current_segment_features:
                    features_sequence = np.array(current_segment_features)
                    pred_result = processor.predict_segment(features_sequence)

                    l3.append(pred_result['top_prediction'].split('.')[-1])
                    
                    if pred_result:
                        predicted_words.append(pred_result['top_prediction'])
                        segment_predictions.append({
                            'segment': len(segments) + 1,
                            'frames': (start_frame, frame_count - pause_duration_threshold - 1),
                            'prediction': pred_result
                        })
                        current_segment_prediction = pred_result
                    
                    segments.append((start_frame, frame_count - pause_duration_threshold - 1))
                pause_start_frame = frame_count - pause_duration_threshold
                current_segment_features = []
            state = "PAUSE"
        else:
            if state == "PAUSE":
                start_frame = pause_start_frame
            state = "RUNNING"
        
        display_frame = frame.copy()
        
        current_word = "Current Word: None"
        if current_segment_prediction:
            current_word = f"Current Word: {current_segment_prediction['top_prediction']}"
        
        sentence = "ISL Sentence: " + " ".join(predicted_words)
        eng_sentence = "English Sentence: " + temp_eng_sentence

        
        cv2.putText(display_frame, current_word, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
        cv2.putText(display_frame, sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)         
        cv2.putText(display_frame, eng_sentence, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)   

        
        cv2.imshow('Sign Language Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break
        
        prev_boxes = current_boxes
    
    if state == "RUNNING" and start_frame is not None and current_segment_features:
        features_sequence = np.array(current_segment_features)
        pred_result = processor.predict_segment(features_sequence)
        if pred_result:
            predicted_words.append(pred_result['top_prediction'])
            segment_predictions.append({
                'segment': len(segments) + 1,
                'frames': (start_frame, frame_count - 1),
                'prediction': pred_result
            })
        segments.append((start_frame, frame_count - 1))
    
    cap.release()
    cv2.destroyAllWindows()
    
    return {
        'segments': segments,
        'predictions': segment_predictions,
        'sentence': " ".join(predicted_words),
        'eng_sentence': temp_eng_sentence
    }









def ensure_directory(directory):

    """Create directory if it doesn't exist"""

    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def visualize_hand_trajectories(video_path, output_folder, processor):
    
    """
    Visualize hand trajectories for each detected segment
    """

    output_dir = ensure_directory(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")


    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    
    for i, (start_frame, end_frame) in enumerate(segments):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        left_hand_x, left_hand_y = [], []
        right_hand_x, right_hand_y = [], []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = processor.holistic.process(image_rgb)
            
            if results.left_hand_landmarks:
                wrist = results.left_hand_landmarks.landmark[0]
                left_hand_x.append(wrist.x)
                left_hand_y.append(wrist.y)
                
            if results.right_hand_landmarks:
                wrist = results.right_hand_landmarks.landmark[0]
                right_hand_x.append(wrist.x)
                right_hand_y.append(wrist.y)
        
        if left_hand_x:
            ax.plot(left_hand_x, left_hand_y, 'b-', label='Left Hand', alpha=0.7)
            ax.scatter(left_hand_x[0], left_hand_y[0], color='blue', s=100, marker='o')
            ax.scatter(left_hand_x[-1], left_hand_y[-1], color='blue', s=100, marker='x')
            
        if right_hand_x:
            ax.plot(right_hand_x, right_hand_y, 'r-', label='Right Hand', alpha=0.7)
            ax.scatter(right_hand_x[0], right_hand_y[0], color='red', s=100, marker='o')
            ax.scatter(right_hand_x[-1], right_hand_y[-1], color='red', s=100, marker='x')
            
        ax.set_title(f'Hand Trajectories for Segment {i+1}')
        ax.set_xlabel('X coordinate (normalized)')
        ax.set_ylabel('Y coordinate (normalized)')
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0) 
        ax.legend()
        ax.grid(True)
        
        plt.savefig(os.path.join(output_dir, f'segment_{i+1}_trajectory.png'))
        plt.close(fig)
    
    cap.release()



def visualize_segmentation_metrics(video_path, output_folder, processor):
   
    """
    Visualize metrics used for segmentation (pause detection, proximity, etc.)
    """

    output_dir = ensure_directory(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
        
    frame_indices = []
    pause_counters_left = []
    pause_counters_right = []
    hand_distances = []
    hand_bottom_positions = []
    segmentation_points = []
    
    prev_boxes = [None, None]
    pause_counters = [0, 0]
    frame_count = 0
    segments = []
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proximity_threshold = 0.2 * frame_width
    bottom_threshold = frame_height * 0.7
    iou_threshold = 0.85
    pause_duration_threshold = 8
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_indices.append(frame_count)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = processor.holistic.process(image_rgb)
        
        current_boxes = get_hand_boxes(results, frame.shape)
        
        for i in range(2):
            if prev_boxes[i] is not None and current_boxes[i] is not None:
                iou, _ = calculate_box_metrics(current_boxes[i], prev_boxes[i])
                if iou > iou_threshold:
                    pause_counters[i] += 1
                else:
                    pause_counters[i] = 0
            else:
                pause_counters[i] = 0
        
        pause_counters_left.append(pause_counters[0])
        pause_counters_right.append(pause_counters[1])
        
        distance = float('inf')
        bottom_pos = 0
        
        if all(current_boxes):
            _, distance = calculate_box_metrics(current_boxes[0], current_boxes[1])
            bottom_pos = max(current_boxes[0][3], current_boxes[1][3]) / frame_height
        
        hand_distances.append(distance)
        hand_bottom_positions.append(bottom_pos)
        
        both_hands_paused = all(pc >= pause_duration_threshold for pc in pause_counters)
        hands_in_proximity = distance < proximity_threshold
        hands_in_bottom = (current_boxes[0][3] > bottom_threshold and 
                         current_boxes[1][3] > bottom_threshold) if all(current_boxes) else False
        
        if both_hands_paused and hands_in_proximity and hands_in_bottom:
            segmentation_points.append(frame_count)
            segments.append(frame_count)
        
        prev_boxes = current_boxes
    
    cap.release()
    
    hand_distances = np.array(hand_distances)
    max_distance = np.max(hand_distances[hand_distances < float('inf')])
    hand_distances = np.minimum(hand_distances, max_distance) / max_distance
    
    window_size = min(51, len(hand_distances) - 1)
    if window_size % 2 == 0:
        window_size -= 1
    
    if len(hand_distances) > window_size:
        hand_distances_smooth = savgol_filter(hand_distances, window_size, 3)
    else:
        hand_distances_smooth = hand_distances
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(frame_indices, pause_counters_left, 'b-', label='Left Hand')
    ax1.plot(frame_indices, pause_counters_right, 'r-', label='Right Hand')
    ax1.axhline(y=pause_duration_threshold, color='g', linestyle='--', label=f'Threshold ({pause_duration_threshold})')
    for seg in segmentation_points:
        ax1.axvline(x=seg, color='m', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Pause Counter')
    ax1.set_title('Hand Pause Detection')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(frame_indices, hand_distances_smooth, 'g-')
    ax2.axhline(y=proximity_threshold/max_distance, color='r', linestyle='--', label=f'Proximity Threshold')
    for seg in segmentation_points:
        ax2.axvline(x=seg, color='m', linestyle=':', alpha=0.7)
    ax2.set_ylabel('Normalized Distance')
    ax2.set_title('Hand Proximity')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(frame_indices, hand_bottom_positions, 'y-')
    ax3.axhline(y=bottom_threshold/frame_height, color='r', linestyle='--', label=f'Bottom Threshold')
    for seg in segmentation_points:
        ax3.axvline(x=seg, color='m', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Relative Position')
    ax3.set_title('Hand Bottom Position')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segmentation_metrics.png'))
    plt.close()
    
    return segments


def visualize_features_pca(video_path, output_folder, processor):
    
    """
    Visualize features using PCA for dimensionality reduction
    """

    from sklearn.decomposition import PCA
    
    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    predictions = results['predictions']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, segment_info in enumerate(predictions):
        start_frame, end_frame = segment_info['frames']
        prediction = segment_info['prediction']['top_prediction']
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        segment_features = []
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = processor.holistic.process(image_rgb)
            
            frame_features = processor.process_frame_for_upper_body(frame, results)
            if frame_features is not None:
                segment_features.append(frame_features)
        
        cap.release()
        
        if not segment_features:
            continue
            
        pca = PCA(n_components=2)
        segment_features_array = np.array(segment_features)
        pca_result = pca.fit_transform(segment_features_array)
        
        ax.scatter(pca_result[:, 0], pca_result[:, 1], label=f'Segment {i+1}: {prediction}')
        
        ax.plot(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        
        ax.scatter(pca_result[0, 0], pca_result[0, 1], marker='o', s=100, edgecolors='black')
        ax.scatter(pca_result[-1, 0], pca_result[-1, 1], marker='x', s=100, edgecolors='black')
    
    ax.set_title('PCA Visualization of Sign Features')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'features_pca.png'))
    plt.close()



def visualize_landmark_heatmap(video_path, output_folder, processor):
   
    """
    Create heatmap of landmark positions for each segment
    """

    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    predictions = results['predictions']
    
    for i, segment_info in enumerate(predictions):
        start_frame, end_frame = segment_info['frames']
        prediction = segment_info['prediction']['top_prediction']
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        frame_count = 0
        while True:
            frame_count += 1
            if frame_count > (end_frame - start_frame):
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = processor.holistic.process(image_rgb)
            
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(heatmap, (x, y), 5, 0.1, -1)
                        
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(heatmap, (x, y), 5, 0.1, -1)
        
        cap.release()
        
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        heatmap_colored = plt.cm.jet(heatmap)
        ax.imshow(heatmap_colored, alpha=0.5)
        
        ax.set_title(f'Landmark Heatmap for Sign: {prediction}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'landmark_heatmap_{i+1}.png'))
        plt.close()


def visualize_all_isl_sentence(video_path, output_folder, processor):
    
    """
    Create a combined visualization for the entire ISL sentence
    """

    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    predictions = results['predictions']
    sentence = results['sentence']
    
    sign_frames = []
    signs = []
    
    for pred in predictions:
        start_frame, end_frame = pred['frames']
        mid_frame = (start_frame + end_frame) // 2
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            sign_frames.append(frame)
            signs.append(pred['prediction']['top_prediction'].split('.')[-1])
    
    if not sign_frames:
        return
    
    n_signs = len(sign_frames)
    fig, axes = plt.subplots(1, n_signs, figsize=(4*n_signs, 6))
    
    if n_signs == 1:
        axes = [axes]
    
    for i, (frame, sign, ax) in enumerate(zip(sign_frames, signs, axes)):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(sign)
        ax.axis('off')
    
    plt.suptitle(f"ISL Sentence: {' '.join(signs)}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'isl_sentence.png'))
    plt.close()


def segment_video(video_path, processor, visualize=False):
    """Modified version of segment_video that doesn't show UI and returns results only"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    prev_boxes = [None, None]
    pause_counters = [0, 0]
    state = "RUNNING"
    segments = []
    start_frame = None
    frame_count = 0
    pause_start_frame = None
    current_segment_features = []
    predicted_words = []
    segment_predictions = []
    current_segment_prediction = None
    
    proximity_threshold = 0.2 * frame_width
    bottom_threshold = frame_height * 0.7
    iou_threshold = 0.85
    pause_duration_threshold = 8
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = processor.holistic.process(image_rgb)
        
        frame_features = processor.process_frame_for_upper_body(frame, results)
        if frame_features is not None:
            current_segment_features.append(frame_features)
        
        current_boxes = get_hand_boxes(results, frame.shape)
        
        for i in range(2):
            if prev_boxes[i] is not None and current_boxes[i] is not None:
                iou, _ = calculate_box_metrics(current_boxes[i], prev_boxes[i])
                if iou > iou_threshold:
                    pause_counters[i] += 1
                else:
                    pause_counters[i] = 0
            else:
                pause_counters[i] = 0
        
        both_hands_paused = all(pc >= pause_duration_threshold for pc in pause_counters)
        hands_in_proximity = False
        hands_in_bottom = False
        
        if all(current_boxes):
            _, distance = calculate_box_metrics(current_boxes[0], current_boxes[1])
            hands_in_proximity = distance < proximity_threshold
            hands_in_bottom = (current_boxes[0][3] > bottom_threshold and 
                             current_boxes[1][3] > bottom_threshold)
        
        if both_hands_paused and hands_in_proximity and hands_in_bottom:
            if state == "RUNNING":
                if start_frame is not None and current_segment_features:
                    features_sequence = np.array(current_segment_features)
                    pred_result = processor.predict_segment(features_sequence)
                    
                    if pred_result:
                        predicted_words.append(pred_result['top_prediction'])
                        segment_predictions.append({
                            'segment': len(segments) + 1,
                            'frames': (start_frame, frame_count - pause_duration_threshold - 1),
                            'prediction': pred_result
                        })
                        current_segment_prediction = pred_result
                    
                    segments.append((start_frame, frame_count - pause_duration_threshold - 1))
                pause_start_frame = frame_count - pause_duration_threshold
                current_segment_features = []
            state = "PAUSE"
        else:
            if state == "PAUSE":
                start_frame = pause_start_frame
            state = "RUNNING"
        
        prev_boxes = current_boxes
    
    if state == "RUNNING" and start_frame is not None and current_segment_features:
        features_sequence = np.array(current_segment_features)
        pred_result = processor.predict_segment(features_sequence)
        if pred_result:
            predicted_words.append(pred_result['top_prediction'])
            segment_predictions.append({
                'segment': len(segments) + 1,
                'frames': (start_frame, frame_count - 1),
                'prediction': pred_result
            })
        segments.append((start_frame, frame_count - 1))
    
    cap.release()
    
    return {
        'segments': segments,
        'predictions': segment_predictions,
        'sentence': " ".join(predicted_words),
        'eng_sentence': ""
    }



def visualize_optical_floww(video_path, output_folder, processor):
   
    """
    Visualize optical flow for each segment, saving both:
    1. Flow visualization with original frame in background
    2. Flow visualization on plain background (just the graph)
    """

    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    for i, (start_frame, end_frame) in enumerate(segments):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, old_frame = cap.read()
        if not ret:
            continue
            
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        if p0 is None:
            cap.release()
            continue
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        color = np.random.randint(0, 255, (100, 3))
        
        mask = np.zeros_like(old_frame)
        
        plain_background = np.zeros_like(old_frame)
        
        frame_count = 0
        while True:
            frame_count += 1
            if frame_count > (end_frame - start_frame):
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            if p1 is None:
                break
                
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j % len(color)].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[j % len(color)].tolist(), -1)
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        img_with_background = cv2.add(old_frame, mask)
        ax1.imshow(cv2.cvtColor(img_with_background, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Optical Flow for Segment {i+1} (with background)')
        ax1.axis('off')
        
        plt.figure(fig1.number)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'optical_flow_segment_{i+1}_with_background.png'))
        
        img_plain = mask.copy()  
        ax2.imshow(cv2.cvtColor(img_plain, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Optical Flow for Segment {i+1} (flow only)')
        ax2.axis('off')
        
        plt.figure(fig2.number)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'optical_flow_segment_{i+1}_flow_only.png'))
        
        plt.close(fig1)
        plt.close(fig2)
        
        cap.release()


def visualize_pure_optical_flow(video_path, output_folder, processor):
   
    """
    Visualize optical flow vectors for the entire video in a single graph without frames,
    focusing on hand movements across all segments
    """

    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if p0 is None:
        cap.release()
        return
    
    flow_data = []
    frame_indices = []
    frame_count = 0
    
    complete_flow_mask = np.zeros_like(first_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    prev_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    color = np.random.randint(0, 255, (100, 3))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_indices.append(frame_count)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        
        avg_magnitude = 0
        
        if p1 is not None and st is not None and len(st) > 0:
            st = st.reshape(-1)
            
            good_indices = np.where(st == 1)[0]
            if len(good_indices) > 0:
                good_new = p1[good_indices]
                good_old = p0[good_indices]
                
                if good_new.shape[1] == 2 and good_old.shape[1] == 2:
                    flows = good_new - good_old
                    magnitudes = np.sqrt(flows[:, 0]**2 + flows[:, 1]**2)
                    
                    if len(magnitudes) > 0:
                        avg_magnitude = np.mean(magnitudes)
                        
                    for j, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        complete_flow_mask = cv2.line(complete_flow_mask, 
                                                     (int(a), int(b)), 
                                                     (int(c), int(d)), 
                                                     color[j % len(color)].tolist(), 2)
        
        flow_data.append(avg_magnitude)
        
        prev_gray = gray.copy()
        
        if p1 is not None and st is not None and np.sum(st) > 0:
            good_indices = np.where(st == 1)[0]
            if len(good_indices) > 0:
                p0 = p1[good_indices].reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        else:
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            
        if p0 is None or len(p0) == 0:
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if p0 is None:
                p0 = np.zeros((1, 1, 2), dtype=np.float32)
    
    cap.release()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frame_indices, flow_data, 'g-', label='Hand Movement Flow Magnitude')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Average Flow Magnitude')
    ax.set_title('Hand Movement Flow Magnitude Over Time')
    ax.grid(True)
    
    for i, (start, end) in enumerate(segments):
        ax.axvspan(start, end, alpha=0.1, color='gray')
    
    if len(flow_data) > 3:  
        window_size = min(21, len(flow_data) - 1)
        if window_size % 2 == 0:
            window_size -= 1
            
        if len(flow_data) > window_size and window_size > 2:
            try:
                from scipy.signal import savgol_filter
                flow_smooth = savgol_filter(flow_data, window_size, 3)
                ax.plot(frame_indices, flow_smooth, 'r-', label='Smoothed Hand Movement', alpha=0.7)
                
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
            except:
                print("Smoothing could not be applied, skipping")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hand_movement_flow_magnitude.png'))
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(cv2.cvtColor(complete_flow_mask, cv2.COLOR_BGR2RGB))
    ax2.set_title('Complete Hand Movement Flow Paths')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complete_hand_flow_paths.png'))
    plt.close()


def visualize_optical_flow(video_path, output_folder, processor):
    
    """
    Visualize optical flow for each segment, focusing on hand movements,
    saving just the flow visualization on plain background
    """

    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    for i, (start_frame, end_frame) in enumerate(segments):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, old_frame = cap.read()
        if not ret:
            continue
            
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        if p0 is None:
            cap.release()
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color = np.random.randint(0, 255, (100, 3))
        
        mask = np.zeros_like(old_frame)
        
        frame_count = 0
        while True:
            frame_count += 1
            if frame_count > (end_frame - start_frame):
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            if p1 is None:
                break
                
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[j % len(color)].tolist(), 2)
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        ax.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Hand Movement Flow for Segment {i+1}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hand_flow_segment_{i+1}.png'))
        plt.close()
        
        cap.release()


def visualize_continuous_optical_flow(video_path, output_folder, processor):
   
    """
    Visualize continuous optical flow for the entire video in a single visualization,
    focusing on hand movements across the entire sentence/video
    """

    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return
    
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    if p0 is None:
        cap.release()
        return
    
    flow_data = []
    frame_indices = []
    
    flow_mask = np.zeros_like(first_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    color = np.random.randint(0, 255, (100, 3))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_indices.append(frame_count)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        
        avg_magnitude = 0
        
        if p1 is not None and st is not None and len(st) > 0:
            st = st.reshape(-1)
            
            good_indices = np.where(st == 1)[0]
            if len(good_indices) > 0:
                good_new = p1[good_indices]
                good_old = p0[good_indices]
                
                if good_new.shape[1] == 2 and good_old.shape[1] == 2:
                    flows = good_new - good_old
                    magnitudes = np.sqrt(flows[:, 0]**2 + flows[:, 1]**2)
                    
                    if len(magnitudes) > 0:
                        avg_magnitude = np.mean(magnitudes)
                        
                    for j, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        flow_mask = cv2.line(flow_mask, 
                                           (int(a), int(b)), 
                                           (int(c), int(d)), 
                                           color[j % len(color)].tolist(), 2)
        
        flow_data.append(avg_magnitude)
        
        old_gray = gray.copy()
        
        if p1 is not None and st is not None and np.sum(st) > 0:
            good_indices = np.where(st == 1)[0]
            if len(good_indices) > 0:
                p0 = p1[good_indices].reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        else:
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            
        if p0 is None or len(p0) == 0:
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if p0 is None:
                p0 = np.zeros((1, 1, 2), dtype=np.float32)
    
    cap.release()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(flow_mask, cv2.COLOR_BGR2RGB))
    ax.set_title('Continuous Hand Movement Flow for Entire Sentence')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'continuous_hand_flow.png'))
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    ax2.plot(frame_indices, flow_data, 'g-', label='Hand Movement Magnitude')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Average Flow Magnitude')
    ax2.set_title('Hand Movement Flow Magnitude Over Entire Sentence')
    ax2.grid(True)
    
    for i, (start, end) in enumerate(segments):
        ax2.axvspan(start, end, alpha=0.1, color='gray')
    
    if len(flow_data) > 3:
        window_size = min(21, len(flow_data) - 1)
        if window_size % 2 == 0:
            window_size -= 1
            
        if len(flow_data) > window_size and window_size > 2:
            try:
                from scipy.signal import savgol_filter
                flow_smooth = savgol_filter(flow_data, window_size, 3)
                ax2.plot(frame_indices, flow_smooth, 'r-', label='Smoothed Movement', alpha=0.7)
                
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys())
            except:
                print("Smoothing could not be applied, skipping")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'continuous_flow_magnitude.png'))
    plt.close()



def visualize_iou_pause_detection(video_path, output_folder, processor, threshold=0.92):
    
    """
    Visualize IoU (Intersection over Union) scores between consecutive frames
    to show how pauses are detected in sign language videos.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save visualizations
        processor: SignLanguageProcessor instance
        threshold: IoU threshold for detecting pauses (default 0.92)
    """

    output_dir = ensure_directory(output_folder)
    
    results = segment_video(video_path, processor, visualize=False)
    segments = results['segments']
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return
    
    iou_scores = []
    frame_indices = []
    pause_frames = []
    segment_boundaries = []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    _, prev_binary = cv2.threshold(prev_gray, 127, 255, cv2.THRESH_BINARY)
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame_indices.append(frame_idx)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        intersection = np.logical_and(prev_binary, binary)
        union = np.logical_or(prev_binary, binary)
        
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        iou_scores.append(iou)
        
        if iou > threshold:
            pause_frames.append(frame_idx)
        
        prev_binary = binary
    
    cap.release()
    
    for start, end in segments:
        segment_boundaries.extend([start, end])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(frame_indices, iou_scores, 'b-', label='IoU Score', alpha=0.7)
    
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    if pause_frames:
        pause_indices = [frame_indices.index(i) for i in pause_frames if i in frame_indices]
        pause_scores = [iou_scores[i] for i in pause_indices]
        ax.scatter([frame_indices[i] for i in pause_indices], pause_scores, 
                  color='red', s=30, alpha=0.5, label='Detected Pauses')
    
    for idx, boundary in enumerate(segment_boundaries):
        if idx % 2 == 0:  
            ax.axvline(x=boundary, color='g', linestyle='-', alpha=0.5, 
                      label='Segment Start' if idx == 0 else "")
        else:  
            ax.axvline(x=boundary, color='orange', linestyle='-', alpha=0.5, 
                      label='Segment End' if idx == 1 else "")
    
    for i, (start, end) in enumerate(segments):
        ax.axvspan(start, end, alpha=0.1, color='gray', 
                 label='Segment Region' if i == 0 else "")
    
    if len(iou_scores) > 3:
        window_size = min(11, len(iou_scores) - 1)
        if window_size % 2 == 0:
            window_size -= 1
            
        if window_size > 2:
            try:
                from scipy.signal import savgol_filter
                iou_smooth = savgol_filter(iou_scores, window_size, 3)
                ax.plot(frame_indices, iou_smooth, 'g-', 
                       label='Smoothed IoU', alpha=0.7)
            except:
                print("Smoothing could not be applied, skipping")
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('IoU Score')
    ax.set_title('Frame Similarity (IoU) for Pause Detection')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])  
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_pause_detection.png'))
    plt.close()
    
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.hist(iou_scores, bins=30, alpha=0.7, color='blue')
    ax1.axvline(x=threshold, color='r', linestyle='--', 
               label=f'Threshold ({threshold})')
    ax1.set_xlabel('IoU Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of IoU Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    sns.heatmap(
        np.array(iou_scores).reshape(1, -1),
        cmap='YlGnBu',
        cbar_kws={'label': 'IoU Score'},
        ax=ax2,
        vmin=0.5,
        vmax=1.0
    )
    
    for boundary in segment_boundaries:
        if boundary < len(iou_scores):
            ax2.axvline(x=boundary, color='r', linestyle='-', alpha=0.7)
    
    ax2.set_title('IoU Scores Over Time (Heatmap)')
    ax2.set_xlabel('Frame Number')
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_distribution.png'))
    plt.close()
    
    print(f"IoU pause detection visualizations saved to {output_dir}")



def create_all_visualizations(video_path, output_folder, processor):
    
    """Generate all visualizations at once"""

    output_dir = ensure_directory(output_folder)
    visualize_hand_trajectories(video_path, output_dir, processor)
    visualize_segmentation_metrics(video_path, output_dir, processor)
    visualize_features_pca(video_path, output_dir, processor)
    visualize_optical_flow(video_path, output_dir, processor)
    visualize_landmark_heatmap(video_path, output_dir, processor)
    visualize_all_isl_sentence(video_path, output_dir, processor)
    visualize_pure_optical_flow(video_path, output_dir, processor)
    visualize_continuous_optical_flow(video_path, output_dir, processor)
    visualize_iou_pause_detection(video_path, output_dir, processor) 
    
    print(f"All visualizations have been saved to {output_dir}")



video_path = "/path//to/isl_video.mp4"  
model_path="/path/to/model.keras"
encoder_path = "/path/to/label_encoder.pkl"

processor = SignLanguageProcessor(model_path, encoder_path)
output_folder = "/path/to/output_dir"
create_all_visualizations(video_path, output_folder, processor)
import os
import sys
import logging
import pickle
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense
import concurrent.futures
from scipy.signal import medfilt
from hybrid_segmentation.combiner import combine_segmentations
from hybrid_segmentation.video_processing import process_video
from tts.deepgram import text_to_speech
import keras

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def process_frame_for_hands_visualization(frame):

    """
    Processes a video frame to detect and visualize hand landmarks using MediaPipe Hands.

    Args:
        frame (numpy.ndarray): The input frame from a video.

    Returns:
        numpy.ndarray: The output frame with hand landmarks drawn.
    """


    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(
        static_image_mode=False,  
        max_num_hands=2,
        min_detection_confidence=0.2
    ) as hands:
        results = hands.process(image_rgb)
    
    output_frame = frame.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                output_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  
            )
    
    return output_frame
    


def visualize_predictions(input_video_path, frame_segments, predictions):

    """
    Visualizes hand tracking and predicted words from a given video.

    Args:
        input_video_path (str): Path to the input video.
        frame_segments (list of tuples): List of (start_frame, end_frame) pairs for word segments.
        predictions (list of str): List of predicted words corresponding to each segment.

    Displays:
        - The original video with predicted words.
        - The video with hand tracking visualization.
    """

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    
    current_segment = 0
    current_prediction = ""
    frame_number = 0
    recognized_words = [] 
    FRAMES_TO_SHOW = 5 
    
    cleaned_predictions = [pred.split('.')[1].strip() if '.' in pred else pred.strip() for pred in predictions]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        current_segment = -1
        for i, (start, end) in enumerate(frame_segments):
            if start <= frame_number < end:
                current_segment = i
                
                frames_remaining = end - frame_number
                if frames_remaining <= FRAMES_TO_SHOW:
                    current_prediction = cleaned_predictions[i]
                    if current_prediction not in recognized_words:
                        recognized_words.append(current_prediction)
                else:
                    current_prediction = ""
                break
            
        hand_tracking_frame = process_frame_for_hands_visualization(frame)
        
        recognized_sentence = " ".join(recognized_words)
        
        debug_info = f"Frame: {frame_number}"
        if current_segment != -1:
            segment_start, segment_end = frame_segments[current_segment]
            frames_remaining = segment_end - frame_number
            debug_info += f" | Segment {current_segment}: {segment_start}-{segment_end}"
        
        for display_frame in [frame, hand_tracking_frame]:
            if current_prediction:
                cv2.putText(display_frame, f"Current word: {current_prediction}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if recognized_words:
                words = recognized_sentence.split()
                lines = []
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    if len(" ".join(current_line)) > 30: 
                        lines.append(" ".join(current_line[:-1]))
                        current_line = [current_line[-1]]
                if current_line:
                    lines.append(" ".join(current_line))
                
                for i, line in enumerate(lines):
                    cv2.putText(display_frame, f"Sentence: {line}", 
                              (10, 80 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            
            cv2.putText(display_frame, debug_info, 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Original Video', frame)
        cv2.imshow('Hand Tracking', hand_tracking_frame)
        
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



def calculate_motion_metrics(hand_landmarks):

    """
    Computes motion metrics such as velocity, acceleration, and jerk from hand landmarks.

    Args:
        hand_landmarks (list of MediaPipe landmarks): List of detected hand landmarks.

    Returns:
        tuple: (average velocity, average acceleration, average jerk).
    """

    if not hand_landmarks:
        return 0, 0, 0
    
    total_metrics = {'velocity': 0, 'acceleration': 0, 'jerk': 0}
    num_hands = len(hand_landmarks)
    
    for landmarks in hand_landmarks:
        points = np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark])
        
        velocities = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
        
        accelerations = np.diff(velocities) if len(velocities) > 1 else np.array([0])
        avg_acceleration = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0
        
        jerk = np.diff(accelerations) if len(accelerations) > 1 else np.array([0])
        avg_jerk = np.mean(np.abs(jerk)) if len(jerk) > 0 else 0
        
        total_metrics['velocity'] += avg_velocity
        total_metrics['acceleration'] += avg_acceleration
        total_metrics['jerk'] += avg_jerk
    
    return (total_metrics['velocity'] / num_hands,
            total_metrics['acceleration'] / num_hands,
            total_metrics['jerk'] / num_hands)


def smooth_metrics(metrics, window_size=5):

    """
    Applies median filtering to smooth motion metrics.

    Args:
        metrics (list of float): Motion metric values.
        window_size (int): The size of the median filter window.

    Returns:
        list: Smoothed metric values.
    """

    if len(metrics) < window_size:
        return metrics
    return medfilt(metrics, window_size)


def draw_enhanced_motion_graph(frame, velocities, accelerations, jerks, max_points=50):

    """
    Draws motion analysis graphs (velocity, acceleration, jerk) on a video frame.

    Args:
        frame (numpy.ndarray): The video frame to overlay graphs on.
        velocities (list of float): Velocity values.
        accelerations (list of float): Acceleration values.
        jerks (list of float): Jerk values.
        max_points (int): Maximum number of data points to plot.

    Returns:
        numpy.ndarray: The output frame with motion graphs overlaid.
    """

    if not velocities or not accelerations:
        return frame
    
    height, width = frame.shape[:2]
    graph_width = width // 3
    graph_height = height // 3
    margin = 20
    x_start = width - graph_width - margin
    y_start = height - graph_height - margin
    
    overlay = frame.copy()
    
    cv2.rectangle(overlay, 
                 (x_start-10, y_start-45), 
                 (x_start+graph_width+10, y_start+graph_height+10),
                 (0, 0, 0),
                 -1)
    
    cv2.putText(overlay,
                "Motion Analysis",
                (x_start, y_start-25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1)
    
    def normalize_data(data, max_points):
        points = data[-max_points:] if len(data) > max_points else data
        max_val = max(max(points), 0.1)
        return [p/max_val for p in points]
    
    vel_smooth = smooth_metrics(velocities[-max_points:])
    acc_smooth = smooth_metrics(accelerations[-max_points:])
    jerk_smooth = smooth_metrics(jerks[-max_points:])
    
    vel_norm = normalize_data(vel_smooth, max_points)
    acc_norm = normalize_data(acc_smooth, max_points)
    jerk_norm = normalize_data(jerk_smooth, max_points)
    
    for i in range(4):
        y_grid = int(y_start + (graph_height * i // 3))
        cv2.line(overlay,
                (x_start, y_grid),
                (x_start + graph_width, y_grid),
                (50, 50, 50),
                1)
    
    def plot_line(y_values, color, label, offset=0):
        for i in range(len(y_values)-1):
            x1 = x_start + int((i/len(y_values)) * graph_width)
            x2 = x_start + int(((i+1)/len(y_values)) * graph_width)
            y1 = y_start + int((1-y_values[i]) * graph_height)
            y2 = y_start + int((1-y_values[i+1]) * graph_height)
            
            cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(overlay,
                   f"{label}: {y_values[-1]:.2f}",
                   (x_start, y_start + graph_height + 15 + offset),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   color,
                   1)
    
    plot_line(vel_norm, (0, 255, 0), "Velocity")
    plot_line(acc_norm, (255, 165, 0), "Acceleration", 15)
    plot_line(jerk_norm, (0, 165, 255), "Jerk", 30)
    
    motion_score = (np.mean(vel_norm) + np.mean(acc_norm) + np.mean(jerk_norm)) / 3
    cv2.putText(overlay,
                f"Motion Score: {motion_score:.2f}",
                (x_start, y_start-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)
    
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    return frame



@keras.saving.register_keras_serializable()
class SelfAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None

    def build(self, input_shape):
        self.query_dense = Dense(self.units, use_bias=False)
        self.key_dense = Dense(self.units, use_bias=False)
        self.value_dense = Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        scores = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        output = tf.matmul(attention_weights, value)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config


def load_trained_model_and_label_encoder(model_path, encoder_path):

    """
    Loads a trained Keras model and its associated label encoder.

    Args:
        model_path (str): Path to the saved Keras model file.
        encoder_path (str): Path to the label encoder pickle file.

    Returns:
        tuple: (Keras model, label encoder) or (None, None) if loading fails.
    """

    try:
        keras.config.enable_unsafe_deserialization()
        
        custom_objects = {
            'SelfAttention': SelfAttention
        }
        
        model = load_model(model_path, custom_objects=custom_objects)

        if model is None:
            print(1)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        return model, label_encoder
    except Exception as e:
        print(f"Error loading model or encoder: {str(e)}")
        return None, None



def process_frame_for_hands(frame):

    """
    Extracts upper body and hand landmarks from a video frame using MediaPipe Holistic.

    Args:
        frame (numpy.ndarray): The input video frame.

    Returns:
        list: A flattened list of extracted features (upper body and hand landmarks).
    """


    mp_holistic = mp.solutions.holistic
    
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        upper_body_landmarks = np.zeros(24)  
        left_hand_landmarks = np.zeros(63)   
        right_hand_landmarks = np.zeros(63)  
        
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,  
            min_detection_confidence=0.4,  
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        ) as holistic:
            results = holistic.process(image_rgb)
            
            if results.pose_landmarks:
                upper_body_points = [11, 12, 13, 14, 15, 16, 23, 24]
                for i, idx in enumerate(upper_body_points):
                    landmark = results.pose_landmarks.landmark[idx]
                    coords = np.array([landmark.x, landmark.y, landmark.z])
                    upper_body_landmarks[i*3:(i+1)*3] = coords
            
            if results.left_hand_landmarks:
                for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                    coords = np.array([landmark.x, landmark.y, landmark.z])
                    left_hand_landmarks[i*3:(i+1)*3] = coords
                    
            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    coords = np.array([landmark.x, landmark.y, landmark.z])
                    right_hand_landmarks[i*3:(i+1)*3] = coords
            
            frame_features = np.concatenate([
                upper_body_landmarks,
                left_hand_landmarks,
                right_hand_landmarks
            ]).tolist()
            
            return frame_features if any(frame_features) else []
            
    except Exception as e:
        logging.error(f"Error in process_frame_for_hands: {str(e)}")
        return [0.0] * 150  

def extract_features_from_frames(frames):
    
    """
    Extracts motion-related features from a sequence of frames.

    Args:
        frames (list of numpy.ndarray): List of video frames.

    Returns:
        numpy.ndarray: Extracted features with shape (1, num_frames, 150).
    """

    video_features = []
    for frame in frames:
        frame_features = process_frame_for_hands(frame)
        if frame_features:
            video_features.append(frame_features)

    if not video_features:
        return np.zeros((1, 1, 150))  
    
    return np.array([video_features])


def process_segment(segment_info, video_path, model, label_encoder, total_frames):
    
    """
    Processes a video segment, extracts features, and makes predictions.

    Args:
        segment_info (tuple): Contains (start_frame, end_frame) and segment index.
        video_path (str): Path to the video file.
        model (keras.Model): Trained model for predictions.
        label_encoder (sklearn.preprocessing.LabelEncoder): Encoder for class labels.
        total_frames (int): Total number of frames in the video.

    Returns:
        tuple: (segment index, predicted class, probabilities, (start_frame, end_frame)).
    """
    start_frame, end_frame = segment_info[0]
    segment_index = segment_info[1]
    end_frame = min(end_frame, total_frames)
    
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        segment_frames = []
        
        for frame_no in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            segment_frames.append(frame)
        
        cap.release()
        
        features = extract_features_from_frames(segment_frames)
        
        if features.shape[1] == 1 and not np.any(features):
            return (segment_index, "No features detected", np.zeros(len(label_encoder.classes_)), (start_frame, end_frame))
        
        predicted_class, probabilities = predict_class(model, label_encoder, features)
        
        return (segment_index, predicted_class, probabilities, (start_frame, end_frame))
        
    except Exception as e:
        logging.error(f"Error in process_segment: {str(e)}")
        return (segment_index, "Error processing segment", np.zeros(len(label_encoder.classes_)), (start_frame, end_frame))


def predict_class(model, label_encoder, features):
    
    """
    Predicts the class of the given feature set using the trained model.

    Args:
        model: Trained machine learning model.
        label_encoder: Label encoder to decode class labels.
        features (numpy.ndarray): Input features of shape (num_frames, 150) or (1, num_frames, 150).

    Returns:
        tuple: (Predicted word (str), Probability distribution (numpy.ndarray)).
    """

    if len(features.shape) == 2:
        features = np.expand_dims(features, axis=0)
    
    prediction = model.predict(features, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)
    probabilities = prediction[0]
    predicted_word = label_encoder.inverse_transform(predicted_class)[0]
    

    predicted_word = predicted_word.split('.')[-1].strip().lower()
    
    return predicted_word, probabilities


def segment_video_and_predict_threaded(input_video_path, frame_segments, model_path, label_encoder_path, max_workers=None):
    
    """
    Processes video segments in parallel and makes predictions.

    Args:
        input_video_path (str): Path to the input video.
        frame_segments (list): List of tuples (start_frame, end_frame) representing segments.
        model_path (str): Path to the trained model.
        label_encoder_path (str): Path to the label encoder file.
        max_workers (int, optional): Maximum number of parallel threads.

    Returns:
        list: List of predicted words for each segment.
    """

    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model, label_encoder = load_trained_model_and_label_encoder(model_path, label_encoder_path)
    
    segment_info = [(segment, i) for i, segment in enumerate(frame_segments)]
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(
                process_segment, 
                segment_info=seg_info, 
                video_path=input_video_path,
                model=model,
                label_encoder=label_encoder,
                total_frames=total_frames
            ): seg_info for seg_info in segment_info
        }
        
        for future in concurrent.futures.as_completed(future_to_segment):
            try:
                segment_index, predicted_class, probabilities, (start_frame, end_frame) = future.result()
                results.append((segment_index, predicted_class, probabilities, start_frame, end_frame))
                
                print(f"Segment {segment_index + 1} (frames {start_frame} to {end_frame}):")
                print(f"Predicted class: {predicted_class}")
                print('Top 5 probabilities:')
                top_indices = np.argsort(probabilities)[-5:][::-1]
                for idx in top_indices:
                    print(f"{label_encoder.classes_[idx]}: {probabilities[idx]:.4f}")
                print("\n")
                
            except Exception as e:
                print(f"Segment processing failed: {str(e)}")
    
    results.sort(key=lambda x: x[0])
    return [result[1] for result in results]


    

def segment_video_and_predict(input_video_path, frame_segments, model_path, label_encoder_path):
    
    """
    Processes a video and makes predictions for each segment with visualization.

    Args:
        input_video_path (str): Path to the input video.
        frame_segments (list): List of tuples (start_frame, end_frame) representing segments.
        model_path (str): Path to the trained model.
        label_encoder_path (str): Path to the label encoder file.

    Returns:
        list: List of predicted words for each segment.
    """

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model, label_encoder = load_trained_model_and_label_encoder(model_path, label_encoder_path)

    if model is None:
        print("1")
    
    predictions = []
    current_prediction = "Not predicted yet"
    combined_sentence = ""
    

    cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Processed View', cv2.WINDOW_NORMAL)
    
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    for i, (start_frame, end_frame) in enumerate(frame_segments):
        end_frame = min(end_frame, total_frames)
        segment_frames = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            for frame_no in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                processed_frame = frame.copy()
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        processed_frame, 
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=1)
                    )
                
                for landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                    if landmarks:
                        mp_drawing.draw_landmarks(
                            processed_frame,
                            landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
                        )
                

                for display_frame in [frame, processed_frame]:
                    cv2.putText(display_frame, f"Frame: {frame_no}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Current: {current_prediction}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Sentence: {combined_sentence}", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2)
                
                cv2.imshow('Original Video', frame)
                cv2.imshow('Processed View', processed_frame)
                
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    break
                
                segment_frames.append(frame)


        features = extract_features_from_frames(segment_frames)
        predicted_class, probabilities = predict_class(model, label_encoder, features)
        
        current_prediction = predicted_class
        predictions.append(predicted_class)
        combined_sentence = " ".join([p.split('.')[1].strip() for p in predictions])
        
        print(f"\nSegment {i+1} (frames {start_frame} to {end_frame}):")
        print(f"Predicted class: {predicted_class}")
        print(f"Combined sentence so far: {combined_sentence}")
        print('Probabilities for each class:')
        for j, prob in enumerate(probabilities):
            print(f"{label_encoder.classes_[j]}: {prob:.4f}")

    cap.release()
    cv2.destroyAllWindows()
    return predictions




def main():

    """
    Main function to process the video, segment it, make predictions,
    and convert the recognized sentence into speech.
    """

    video_path='/path/to/video.mp4'
    model_file = '/path/to/model_file.keras'
    label_encoder_file = '/path/to/label_encoder.pkl'

    video_features = process_video(video_path)
    print(f"Processed video. Shape of features: {video_features.shape}")
    
    y_min = 0.7
    y_max = 1.0
    consecutive_frames = 5
    segments = combine_segmentations(video_features, y_min=y_min, y_max=y_max, consecutive_frames=consecutive_frames)
    print(f"Number of segments: {len(segments)}")
    print(segments)

    # Without threading
    sentence=segment_video_and_predict(video_path, segments, model_file, label_encoder_file)
    print(sentence)
    sentence=[i.split('.')[1].strip() for i in sentence]
    recognized_sentence=" ".join(sentence)
    print(recognized_sentence)
    text_to_speech(recognized_sentence)


    # With threading
    predictions = segment_video_and_predict_threaded(
        video_path, 
        segments, 
        model_file, 
        label_encoder_file,
        max_workers=min(len(segments), 4)  
    )
    print(predictions)

    sentence = [i.split('.')[1].strip() for i in predictions]
    recognized_sentence = " ".join(sentence)
    print("Recognized sentence:", recognized_sentence)
    
    visualize_predictions(video_path, segments, predictions)
    
    text_to_speech(recognized_sentence)



if __name__ == "__main__":
    main()





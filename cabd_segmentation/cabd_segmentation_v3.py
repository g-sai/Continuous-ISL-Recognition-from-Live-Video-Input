import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import keras
from tensorflow import keras

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from translation.translation import translate, initialize_chat
from tts.deepgram import text_to_speech


"""
This segmentation approach pauses for less than a second at the segment boundary to indicate where a segment is created.
The video while being also visualises the upper body and hand landmarks, to understand how the feature extraction works.
"""


# conversation = initialize_chat()



class BiomechanicalSignProcessor:

    """
    A class to process sign language gestures using a trained deep learning model.
    
    Attributes:
        model_path (str): The path to the model file.
        encoder_path (str) : The path to label encoder file.
    """

    def __init__(self, model_path, encoder_path):

        self.model, self.label_encoder = self.load_model_and_encoder(model_path, encoder_path)
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,  
            model_complexity=1,  
            enable_segmentation=False,  
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
            refine_face_landmarks=False  
        )
        

        self.kinematic_body_indices = np.array([11, 12, 13, 14, 15, 16, 23, 24])
        
        self.upper_kinematic_features = np.zeros(24)
        self.left_hand_kinematic_features = np.zeros(63)
        self.right_hand_kinematic_features = np.zeros(63)

    
    
    def draw_landmarks(self, frame, results):

        """
        Draws landmarks for hands and upper body on the given frame.

        Args:
            frame (np.ndarray): The image frame on which the landmarks will be drawn.
            results (object): The results object containing landmarks data for hands and the upper body.

        """

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.pose_landmarks:
            upper_body_connections = [
                (11, 12), 
                (11, 13), (13, 15),  
                (12, 14), (14, 16),  
                (11, 23), (12, 24),  
                (23, 24)  
            ]
            
            for idx in self.kinematic_body_indices:
                landmark = results.pose_landmarks.landmark[idx]
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            for connection in upper_body_connections:
                start_idx, end_idx = connection
                start_landmark = results.pose_landmarks.landmark[start_idx]
                end_landmark = results.pose_landmarks.landmark[end_idx]
                
                h, w, _ = frame.shape
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)



    def load_model_and_encoder(self, model_path, encoder_path):
        
        """
        Load the sign language recognition model and label encoder.

        Args:
            model_path (str): Path to the trained Keras model file.
            encoder_path (str): Path to the label encoder pickle file.

        Returns:
            tuple: (model, label_encoder) if successful, else (None, None).
        """

        try:
            keras.config.enable_unsafe_deserialization()
            model = load_model(model_path)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            return model, label_encoder
        except Exception as e:
            print(f"Error loading model or encoder: {str(e)}")
            return None, None


    def process_kinematic_features(self, frame, results):
        
        """
        Extract upper body and hand landmarks from a frame.

        Args:
            frame (np.array): The current video frame.
            results (mp.solutions.holistic.HolisticResults): MediaPipe holistic model output.

        Returns:
            np.array: A feature vector containing extracted kinematic features.
        """     

        try:
            self.upper_kinematic_features.fill(0)
            self.left_hand_kinematic_features.fill(0)
            self.right_hand_kinematic_features.fill(0)
            
            if results.pose_landmarks:
                for i, idx in enumerate(self.kinematic_body_indices):
                    landmark = results.pose_landmarks.landmark[idx]
                    self.upper_kinematic_features[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]
            
            if results.left_hand_landmarks:
                for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                    self.left_hand_kinematic_features[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]
            
            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    self.right_hand_kinematic_features[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]
            
            return np.concatenate([
                self.upper_kinematic_features,
                self.left_hand_kinematic_features,
                self.right_hand_kinematic_features
            ])
        
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return np.zeros(150)

    
            
    def predict_segment(self, features, confidence_threshold=0.1):
        
        """
        Predicts a sign language segment based on extracted features.

        Args:
            features (np.array): The extracted kinematic feature sequence.
            confidence_threshold (float): Minimum confidence for accepting a prediction.

        Returns:
            dict: A dictionary containing the predicted sign and confidence score.
        """
        
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


def extract_hand_regions(results, image_shape):

    """
    Extract bounding boxes of hands from the MediaPipe results.

    Args:
        results (mp.solutions.holistic.HolisticResults): Holistic model output.
        image_shape (tuple): The shape of the image (height, width).

    Returns:
        list: Bounding box coordinates for left and right hands.
    """
    
    regions = [None, None]  
    
    if results.left_hand_landmarks:
        landmarks = np.array([[lm.x * image_shape[1], lm.y * image_shape[0]] 
                            for lm in results.left_hand_landmarks.landmark])
        padding = 20
        min_coords = landmarks.min(axis=0)
        max_coords = landmarks.max(axis=0)
        
        regions[0] = (
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
        
        regions[1] = (
            max(0, int(min_coords[0]) - padding),
            max(0, int(min_coords[1]) - padding),
            min(image_shape[1], int(max_coords[0]) + padding),
            min(image_shape[0], int(max_coords[1]) + padding)
        )
    
    return regions


def calculate_kinematic_metrics(region1, region2):
    
    """
    Compute the Intersection over Union (IoU) and Euclidean distance between two bounding boxes.

    Args:
        region1 (tuple): Bounding box coordinates (x1, y1, x2, y2).
        region2 (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: (IoU score, Euclidean distance between centers).
    """
    
    if not all(region1) or not all(region2):
        return 0, float('inf')
    
    center1 = [(region1[0] + region1[2]) / 2, (region1[1] + region1[3]) / 2]
    center2 = [(region2[0] + region2[2]) / 2, (region2[1] + region2[3]) / 2]
    
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    xA = max(region1[0], region2[0])
    yA = max(region1[1], region2[1])
    xB = min(region1[2], region2[2])
    yB = min(region1[3], region2[3])
    
    if xB <= xA or yB <= yA:
        return 0, distance
    
    interArea = (xB - xA) * (yB - yA)
    region1Area = (region1[2] - region1[0]) * (region1[3] - region1[1])
    region2Area = (region2[2] - region2[0]) * (region2[3] - region2[1])
    
    if region1Area <= 0 or region2Area <= 0:
        return 0, distance
    
    kinematic_rest = interArea / float(region1Area + region2Area - interArea)
    return max(0, min(1, kinematic_rest)), distance



def analyze_sign_sequence(video_path, processor, kinematic_rest_threshold=0.85, rest_duration_threshold=8, bilateral_sync_threshold=0.2):
    
    """
    Analyze a video sequence to segment and predict sign language gestures without visualizing landmarks.

    Args:
        video_path (str): Path to the input video file.
        processor (BiomechanicalSignProcessor): An instance of BiomechanicalSignProcessor to process frames.
        kinematic_rest_threshold (float, optional): Threshold for kinematic rest detection (default is 0.85).
        rest_duration_threshold (int, optional): Number of consecutive frames that must be in rest state (default is 8).
        bilateral_sync_threshold (float, optional): Threshold for bilateral hand synchronization (default is 0.2).

    Returns:
        dict: A dictionary containing:
            - 'segments': List of tuples (start_frame, end_frame) for each detected segment.
            - 'predictions': List of dictionaries with segment analysis (segment number, frame range, prediction details).
            - 'sentence': The accumulated ISL sentence (string) from the detected segments.
            - 'eng_sentence': The corresponding English translation (string).
    """   
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    prev_hand_regions = [None, None]
    kinematic_rest_counters = [0, 0]
    signing_state = "ACTIVE"
    detected_segments = []
    segment_start_frame = None
    frame_count = 0
    rest_start_frame = None
    current_kinematic_features = []
    detected_signs = []
    segment_analysis = []
    current_segment_result = None

    temp_eng_structure = []
    temp_eng_translation = ''
    
    bilateral_sync_distance = bilateral_sync_threshold * frame_width
    neutral_space_threshold = frame_height * 0.7
    
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = processor.holistic.process(image_rgb)

        processor.draw_landmarks(frame, results)
        
        frame_kinematics = processor.process_kinematic_features(frame, results)
        if frame_kinematics is not None:
            current_kinematic_features.append(frame_kinematics)
        
        current_hand_regions = extract_hand_regions(results, frame.shape)
        
        for i in range(2):
            if prev_hand_regions[i] is not None and current_hand_regions[i] is not None:
                kinematic_rest, _ = calculate_kinematic_metrics(current_hand_regions[i], prev_hand_regions[i])
                if kinematic_rest > kinematic_rest_threshold:
                    kinematic_rest_counters[i] += 1
                else:
                    kinematic_rest_counters[i] = 0
            else:
                kinematic_rest_counters[i] = 0
        
        kinematic_rest_detected = all(pc >= rest_duration_threshold for pc in kinematic_rest_counters)
        bilateral_sync_detected = False
        neutral_space_detected = False
        
        if all(current_hand_regions):
            _, distance = calculate_kinematic_metrics(current_hand_regions[0], current_hand_regions[1])
            bilateral_sync_detected = distance < bilateral_sync_distance
            neutral_space_detected = (current_hand_regions[0][3] > neutral_space_threshold and 
                                   current_hand_regions[1][3] > neutral_space_threshold)
        
        if kinematic_rest_detected and bilateral_sync_detected and neutral_space_detected:
            if signing_state == "ACTIVE":
                if segment_start_frame is not None and current_kinematic_features:
                    kinematics_sequence = np.array(current_kinematic_features)
                    pred_result = processor.predict_segment(kinematics_sequence)

                    temp_eng_structure.append(pred_result['top_prediction'].split('.')[-1])
                    # temp_eng_translation=translate(" ".join(temp_eng_structure),conversation)
                    
                    if pred_result:
                        detected_signs.append(pred_result['top_prediction'])
                        segment_analysis.append({
                            'segment': len(detected_segments) + 1,
                            'frames': (segment_start_frame, frame_count - rest_duration_threshold - 1),
                            'prediction': pred_result
                        })
                        current_segment_result = pred_result
                    
                    detected_segments.append((segment_start_frame, frame_count - rest_duration_threshold - 1))
                rest_start_frame = frame_count - rest_duration_threshold
                current_kinematic_features = []
            signing_state = "REST"
        else:
            if signing_state == "REST":
                segment_start_frame = rest_start_frame
            signing_state = "ACTIVE"
        
        display_frame = frame.copy()
        
        current_sign = "Current Sign: None"
        if current_segment_result:
            current_sign = f"Current Sign: {current_segment_result['top_prediction']}"
        
        isl_sequence = "ISL Sequence: " + " ".join(detected_signs)
        eng_translation = "English Translation: " + temp_eng_translation

        
        cv2.putText(display_frame, current_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)   # Cyan for current word
        cv2.putText(display_frame, isl_sequence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)         # Yellow for accumulated sentence
        cv2.putText(display_frame, eng_translation, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)     # Magenta for converted sentence

        
        cv2.imshow('Sign Language Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break
        
        prev_hand_regions = current_hand_regions
    

    if signing_state == "ACTIVE" and segment_start_frame is not None and current_kinematic_features:
        kinematics_sequence = np.array(current_kinematic_features)
        pred_result = processor.predict_segment(kinematics_sequence)
        if pred_result:
            detected_signs.append(pred_result['top_prediction'])
            segment_analysis.append({
                'segment': len(detected_segments) + 1,
                'frames': (segment_start_frame, frame_count - 1),
                'prediction': pred_result
            })
        detected_segments.append((segment_start_frame, frame_count - 1))
    
    cap.release()
    cv2.destroyAllWindows()
    
    return {
        'segments': detected_segments,
        'predictions': segment_analysis,
        'sentence': " ".join(detected_signs),
        'eng_sentence': temp_eng_translation
    }


def main():

    """
    Main function to analyze a sign language video sequence.
    """

    video_path = "/path/to/video.mp4"  
    model_path="/path/to/model.keras"
    encoder_path = "/path/to/label_encoder.pkl"   

    #t-expensive
    # video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/ga,hru,t.mp4"  
    # model_path="/Users/aswath/Desktop/ISL/review1/model_new_review2_new/epoch_models/model_epoch_085.keras"
    # encoder_path = "/Users/aswath/Desktop/ISL/review1/model_new_review2/label_encoder.pkl"

    #c-loud
    # video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/p,h,c,b.mp4"  
    # model_path="/Users/aswath/Desktop/ISL/review1/model_new_review2_new/epoch_models/model_epoch_085.keras"
    # encoder_path = "/Users/aswath/Desktop/ISL/review1/model_new_review2/label_encoder.pkl"

    # video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/i,g,s,u.mp4"  
    # model_path="/Users/aswath/Desktop/ISL/review1/model_new_review2_new/epoch_models/model_epoch_085.keras"
    # encoder_path = "/Users/aswath/Desktop/ISL/review1/model_new_review2/label_encoder.pkl"

    # video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/h,f,a.mp4"  
    # model_path="/Users/aswath/Desktop/ISL/review1/model_new_review2_new/epoch_models/model_epoch_085.keras"
    # encoder_path = "/Users/aswath/Desktop/ISL/review1/model_new_review2/label_encoder.pkl"

    # video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/if,f,m,s,gf.mp4"  
    # model_path="/Users/aswath/Desktop/ISL/review1/model_new_review2_new/epoch_models/model_epoch_085.keras"
    # encoder_path = "/Users/aswath/Desktop/ISL/review1/model_new_review2/label_encoder.pkl"

    # video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/h,ga,hru,iyh.mp4"  
    # model_path="/Users/aswath/Desktop/ISL/review1_models/Model/fine_tuned_model.keras"
    # encoder_path = "/Users/aswath/Desktop/ISL/review1_models/Model/updated_label_encoder.pkl"
    
    processor = BiomechanicalSignProcessor(model_path, encoder_path)
    results = analyze_sign_sequence(video_path, processor)
    # text_to_speech(results['eng_sentence'])
    
    print("\nAnalysis Results:")
    print(f"Detected Sign Segments: {len(results['segments'])}")
    print(f"ISL Sequence: {results['sentence']}")
    print(f"English Translation: {results['eng_sentence']}")
    
    print("\nSegment Analysis:")
    for pred in results['predictions']:
        print(f"\nSegment {pred['segment']}:")
        print(f"Frames: {pred['frames']}")
        print(f"Prediction: {pred['prediction']['top_prediction']}")
        print(f"Confidence: {pred['prediction']['confidence']*100:.1f}%")


if __name__ == "__main__":
    main()
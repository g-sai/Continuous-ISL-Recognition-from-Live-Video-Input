import cv2
import numpy as np
import re
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
This final CABD segmentation approach doesn't pause to represent natural flow of the signer.
This has been enhanced to provide support for live video as well.
"""


conversation = initialize_chat()


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
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  
            enable_segmentation=False,
            min_detection_confidence=0.3,  
            min_tracking_confidence=0.3,
            smooth_landmarks=False,  
            refine_face_landmarks=False
        )
        
        self.kinematic_body_indices = np.array([11, 12, 13, 14, 15, 16, 23, 24])
        
        self.upper_kinematic_features = np.zeros(24, dtype=np.float32)
        self.left_hand_kinematic_features = np.zeros(63, dtype=np.float32)
        self.right_hand_kinematic_features = np.zeros(63, dtype=np.float32)
        
        self.prediction_queue = []
        
        self.model.predict(np.zeros((1, 1, 150), dtype=np.float32))


    def predict_segment_async(self, kinematic_features):
        
        """
        Queue kinematic features for asynchronous prediction.

        Args:
            kinematic_features (np.ndarray): The kinematic features for the current segment.
        
        """

        self.prediction_queue.append(kinematic_features)
        
    
    def process_prediction_queue(self, confidence_threshold=0.1):
        
        """
        Process the queued prediction without blocking.
        
        Args:
            confidence_threshold (float, optional): The minimum confidence required to accept the prediction (default is 0.1).
        
        Returns:
            dict : A dictionary with keys 'top_prediction' and 'confidence' if prediction is successful
        """        

        if not self.prediction_queue:
            return None
            
        kinematic_features = self.prediction_queue.pop(0)
        try:
            features_array = np.array(kinematic_features)
            
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
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                self.upper_kinematic_features = landmarks[self.kinematic_body_indices].ravel()
            
            if results.left_hand_landmarks:
                self.left_hand_kinematic_features = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).ravel()
            
            if results.right_hand_landmarks:
                self.right_hand_kinematic_features = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).ravel()
            
            return np.concatenate([
                self.upper_kinematic_features,
                self.left_hand_kinematic_features,
                self.right_hand_kinematic_features
            ])
        except Exception as e:
            return np.zeros(150, dtype=np.float32)


def analyze_sign_sequence_ui(video_path, processor, kinematic_rest_threshold=0.85, rest_duration_threshold=8, bilateral_sync_threshold=0.2, ui_update_callback=None, headless=False):

    """
    Segment video using the CABD method with UI updates for streamlit.

    Args:
        video_path (str): Path to the input video file.
        processor (BiomechanicalSignProcessor): An instance that extracts kinematic features and performs predictions.
        kinematic_rest_threshold (float, optional): Threshold for detecting rest in hand movement (default: 0.85).
        rest_duration_threshold (int, optional): Number of consecutive frames in rest state to trigger segmentation (default: 8).
        bilateral_sync_threshold (float, optional): Threshold for bilateral hand synchronization (default: 0.2).
        ui_update_callback (callable, optional): A callback function that receives updated ISL sequence, English translation, and optionally the frame for UI display.
        headless (bool, optional): If True, the function runs without displaying frames (default: False).

    Returns:
        dict: A dictionary containing:
            - 'segments': List of tuples representing the start and end frames of detected segments.
            - 'predictions': List of dictionaries with segment analysis (segment number, frame range, prediction details).
            - 'sentence': A string representing the accumulated ISL sequence.
            - 'eng_sentence': A string representing the translated English sentence.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
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
    pending_prediction = False

    sign_sequence = []
    temp_eng_translation = ''
    temp_eng_structure = []
    
    bilateral_sync_distance = bilateral_sync_threshold * frame_width
    neutral_space_threshold = frame_height * 0.7
    
    if not headless:
        try:
            cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Sign Language Recognition', 800, 600)
        except cv2.error as e:
            print(f"Warning: Could not create display window. Error: {str(e)}. Running in headless mode.")
            headless = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            if processor.prediction_queue:
                pred_result = processor.process_prediction_queue()
                if pred_result:
                    sign_word = pred_result['top_prediction'].split('.')[-1]
                    temp_eng_structure.append(sign_word)
                    
                    detected_signs.append(pred_result['top_prediction'])
                    sign_sequence.append(sign_word)
                    
                    temp_eng_translation=translate(" ".join(temp_eng_structure),conversation)
                    
                    segment_analysis.append({
                        'segment': len(detected_segments) + 1,
                        'frames': (segment_start_frame, frame_count - rest_duration_threshold - 1),
                        'prediction': pred_result
                    })
                    current_segment_result = pred_result
                    
                    if ui_update_callback:
                        isl_sequence = " ".join(detected_signs)
                        
                        #for streamlit-v1
                        ui_update_callback(isl_sequence, temp_eng_translation)
                        #for streamlit-v2
                        # ui_update_callback(isl_sequence, temp_eng_translation, frame)
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = processor.holistic.process(image_rgb)
            
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
                        processor.predict_segment_async(kinematics_sequence)
                        detected_segments.append((segment_start_frame, frame_count - rest_duration_threshold - 1))
                    rest_start_frame = frame_count - rest_duration_threshold
                    current_kinematic_features = []
                signing_state = "REST"
            else:
                if signing_state == "REST":
                    segment_start_frame = rest_start_frame
                signing_state = "ACTIVE"
            
            if not headless:
                try:
                    display_frame = frame.copy()
                    
                    current_sign = "Current Sign: None"
                    if current_segment_result:
                        current_sign = f"Current Sign: {current_segment_result['top_prediction']}"
                    
                    isl_sequence = " ".join(detected_signs)
                    
                    cv2.putText(display_frame, current_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"ISL Sequence: {isl_sequence}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(display_frame, f"English Translation: {temp_eng_translation}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    cv2.imshow('Sign Language Recognition', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  
                        break
                    
                    if ui_update_callback and frame_count % 5 == 0:  
                        isl_sequence = " ".join(detected_signs)
                        ui_update_callback(isl_sequence, temp_eng_translation, display_frame)
                        
                except cv2.error as e:
                    print(f"Warning: Display error occurred. Error: {str(e)}. Continuing in headless mode.")
                    headless = True
            
            prev_hand_regions = current_hand_regions

    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1) 
            except cv2.error:
                pass
    
    while processor.prediction_queue:
        pred_result = processor.process_prediction_queue()
        if pred_result:
            clean_word = re.sub(r'^\d+\.\s*', '', pred_result['top_prediction'])
            sign_sequence.append(clean_word)
            segment_analysis.append({
                'segment': len(detected_segments) + 1,
                'frames': (segment_start_frame, frame_count - 1),
                'prediction': clean_word
            })
            
            if ui_update_callback:
                isl_sequence = " ".join(detected_signs)
                temp_eng_translation = " ".join(temp_eng_structure)
                ui_update_callback(isl_sequence, temp_eng_translation, None)
                
        detected_segments.append((segment_start_frame, frame_count - 1))
    
    return {
        'segments': detected_segments,
        'predictions': segment_analysis,
        'sentence': " ".join(detected_signs),
        'eng_sentence': temp_eng_translation
    }



def analyze_sign_sequence(video_path, processor, kinematic_rest_threshold=0.85, rest_duration_threshold=8, bilateral_sync_threshold=0.2,):

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

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
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
    pending_prediction = False

    sign_sequence = []
    temp_eng_translation = ''

    temp_eng_structure = []

    
    bilateral_sync_distance = bilateral_sync_threshold * frame_width
    neutral_space_threshold = frame_height * 0.7
    
    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if processor.prediction_queue:
            pred_result = processor.process_prediction_queue()
            if pred_result:
                temp_eng_structure.append(pred_result['top_prediction'].split('.')[-1])
                temp_eng_translation = translate(" ".join(temp_eng_structure),conversation)
                detected_signs.append(pred_result['top_prediction'])
                sign_sequence.append(pred_result['top_prediction'].split('.')[-1])
                segment_analysis.append({
                    'segment': len(detected_segments) + 1,
                    'frames': (segment_start_frame, frame_count - rest_duration_threshold - 1),
                    'prediction': pred_result
                })
                current_segment_result = pred_result
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = processor.holistic.process(image_rgb)
        
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
                    processor.predict_segment_async(kinematics_sequence)
                    # temp_eng_translation=translate()
                    
                    detected_segments.append((segment_start_frame, frame_count - rest_duration_threshold - 1))
                rest_start_frame = frame_count - rest_duration_threshold
                current_kinematic_features = []
            signing_state = "REST"
        else:
            if signing_state == "REST":
                segment_start_frame = rest_start_frame
            signing_state = "ACTIVE"
        
        if frame_count % 3 == 0:
            display_frame = frame.copy()
            
            current_sign = "Current Sign: None"
            if current_segment_result:
                current_sign = f"Current Sign: {current_segment_result['top_prediction']}"
            
            isl_sequence = "ISL Sequence: " + " ".join(detected_signs)
            eng_translation = "English Translation: " + temp_eng_translation

            cv2.putText(display_frame, current_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, isl_sequence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, eng_translation, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow('Sign Language Recognition', display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        prev_hand_regions = current_hand_regions
        
    
    while processor.prediction_queue:
        pred_result = processor.process_prediction_queue()
        if pred_result:
            detected_signs.append(pred_result['top_prediction'])
            sign_sequence.append(pred_result['top_prediction'].split('.')[-1])
            segment_analysis.append({
                'segment': len(detected_segments) + 1,
                'frames': (segment_start_frame, frame_count - 1),
                'prediction': pred_result
            })
    
    cap.release()
    cv2.destroyAllWindows()
    
    return {
        'segments': detected_segments,
        'predictions': segment_analysis,
        'sentence': " ".join(detected_signs),
        'eng_sentence': temp_eng_translation

    }



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




def process_live_video(processor, kinematic_rest_threshold=0.85,rest_duration_threshold=8, bilateral_sync_threshold=0.2, ui_update_callback=None):
    
    """
    Process live video feed from the webcam for sign language detection and translation.

    Args:
        processor (BiomechanicalSignProcessor): An instance to process frames and perform predictions.
        kinematic_rest_threshold (float, optional): Threshold for kinematic rest detection (default: 0.85).
        rest_duration_threshold (int, optional): Number of consecutive frames required for a rest state (default: 8).
        bilateral_sync_threshold (float, optional): Threshold for hand synchronization (default: 0.2).
        ui_update_callback (callable, optional): Function to update the UI with the current ISL sequence and translation.

    Returns:
        dict: A dictionary containing:
            - 'segments': List of detected segment frame intervals.
            - 'predictions': List of segment analysis details.
            - 'sentence': The accumulated ISL sentence.
            - 'eng_sentence': The translated English sentence.
    """

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open webcam.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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

    sign_sequence = []
    temp_eng_translation = ''
    temp_eng_structure = []
    
    bilateral_sync_distance = bilateral_sync_threshold * frame_width
    neutral_space_threshold = frame_height * 0.7
    
    cv2.namedWindow('Live Sign Language Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Sign Language Recognition', 800, 600)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            if processor.prediction_queue:
                pred_result = processor.process_prediction_queue()
                if pred_result:
                    temp_eng_structure.append(pred_result['top_prediction'].split('.')[-1])
                    temp_eng_translation = translate(" ".join(temp_eng_structure), conversation)
                    detected_signs.append(pred_result['top_prediction'])
                    sign_sequence.append(pred_result['top_prediction'].split('.')[-1])
                    segment_analysis.append({
                        'segment': len(detected_segments) + 1,
                        'frames': (segment_start_frame, frame_count - rest_duration_threshold - 1),
                        'prediction': pred_result
                    })
                    current_segment_result = pred_result
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = processor.holistic.process(image_rgb)
            
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
                        processor.predict_segment_async(kinematics_sequence)
                        detected_segments.append((segment_start_frame, frame_count - rest_duration_threshold - 1))
                    rest_start_frame = frame_count - rest_duration_threshold
                    current_kinematic_features = []
                signing_state = "REST"
            else:
                if signing_state == "REST":
                    segment_start_frame = rest_start_frame
                signing_state = "ACTIVE"
            
            display_frame = frame.copy()
            
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display_frame, 
                    results.pose_landmarks, 
                    mp.solutions.holistic.POSE_CONNECTIONS)
            
            if results.left_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display_frame, 
                    results.left_hand_landmarks, 
                    mp.solutions.holistic.HAND_CONNECTIONS)
            
            if results.right_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display_frame, 
                    results.right_hand_landmarks, 
                    mp.solutions.holistic.HAND_CONNECTIONS)
            
            current_sign = "Current Sign: None"
            if current_segment_result:
                current_sign = f"Current Sign: {current_segment_result['top_prediction']}"
            
            isl_sequence = "ISL Sequence: " + " ".join(detected_signs)
            eng_translation = "English Translation: " + temp_eng_translation
            
            state_text = f"State: {signing_state}"
            cv2.putText(display_frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.putText(display_frame, current_sign, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, isl_sequence, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, eng_translation, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            if ui_update_callback:
                ui_update_callback(isl_sequence, eng_translation)
            
            cv2.putText(display_frame, "Press 'q' to quit", (10, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Live Sign Language Recognition', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  
                break
            
            prev_hand_regions = current_hand_regions

    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        while processor.prediction_queue:
            pred_result = processor.process_prediction_queue()
            if pred_result:
                temp_eng_structure.append(pred_result['top_prediction'].split('.')[-1])
                detected_signs.append(pred_result['top_prediction'])
                sign_sequence.append(pred_result['top_prediction'].split('.')[-1])
                segment_analysis.append({
                    'segment': len(detected_segments) + 1,
                    'frames': (segment_start_frame, frame_count - 1),
                    'prediction': pred_result
                })
        
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1) 
    
    return {
        'segments': detected_segments,
        'predictions': segment_analysis,
        'sentence': " ".join(detected_signs),
        'eng_sentence': temp_eng_translation
    }


def add_captions_to_video(video_path, segments, predictions, output_video="captioned_output.mp4"):
    
    """
    Overlays real-time captions onto a video based on ISL segmentation and word predictions.

    Parameters:
    - video_path (str): Path to the input video.
    - segments (list): List of (start_frame, end_frame) tuples for each sign segment.
    - predictions (list): List of predicted words corresponding to each segment.
    - output_video (str): Path to save the new video with captions.

    Returns:
    - output_video (str): Path to the final video with embedded captions.
    """

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    current_caption = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        for (segment_start_frame, end_frame), prediction in zip(segments, predictions):
            if frame_count == end_frame:  
                current_caption = prediction

        if current_caption:
            cv2.putText(frame, str(current_caption), (50, height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Captioned video saved as: {output_video}")
    return output_video


def main():

    """
    Main function to analyze a sign language video sequence.
    """
    
    video_path = "/path/to/video.mp4"  
    model_path="/path/to/model.keras"
    encoder_path = "/path/to/label_encoder.pkl" 

    # video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/ga,hru,t.mp4"  
    # model_path="/Users/aswath/Desktop/ISL/review1/model_new_review2_new/epoch_models/model_epoch_085.keras"
    # encoder_path = "/Users/aswath/Desktop/ISL/review1/model_new_review2/label_encoder.pkl"

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
    #15 -29

    video_path = "/Users/aswath/Desktop/ISL/review1/sentence_videos/h,ga,hru,iyh.mp4"  
    model_path="/Users/aswath/Desktop/ISL/review1_models/Model/fine_tuned_model.keras"
    encoder_path = "/Users/aswath/Desktop/ISL/review1_models/Model/updated_label_encoder.pkl"  
    
    processor = BiomechanicalSignProcessor(model_path, encoder_path)
    
    #For recorded video
    results = analyze_sign_sequence(video_path, processor)

    #For live video
    # results = process_live_video(processor)

    text_to_speech(results['eng_sentence'])
    
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



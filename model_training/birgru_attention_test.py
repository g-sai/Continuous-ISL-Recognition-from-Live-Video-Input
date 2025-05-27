import numpy as np
import pickle
import tensorflow as tf
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model


# WARNING: Enabling unsafe deserialization bypasses safety checks and should only be used when you trust the model source.
# The people who developed the model, used this due to a certain system constraint
tf.keras.utils.disable_interactive_logging()
tf.keras.config.enable_unsafe_deserialization()



def load_label_encoder(encoder_path):

    """
    Load a label encoder from a pickle file.
    
    Args:
        encoder_path (str): The file path to the pickle file containing the label encoder.
    
    Returns:
        LabelEncoder: The loaded label encoder.
    """

    with open(encoder_path, 'rb') as f:
        return pickle.load(f)

def process_frame(frame, holistic):

    """
    Process a single frame to extract upper body and hand landmarks using MediaPipe.
    
    Args:
        frame (np.ndarray): The video frame in BGR format.
        holistic (mp.solutions.holistic.Holistic): An instance of the MediaPipe Holistic solution.
    
    Returns:
        tuple: A tuple containing:
            - features (np.ndarray): Concatenated landmark features (upper body, left and right hands).
            - results (mp - landmark_list): The MediaPipe results.
    """

    upper_body_landmarks = np.zeros(24)    # 8 points * 3 coordinates
    left_hand_landmarks = np.zeros(63)     # 21 points * 3 coordinates
    right_hand_landmarks = np.zeros(63)    # 21 points * 3 coordinates
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
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
    
    features = np.concatenate([
        upper_body_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
    ])
    
    return features, results

def draw_landmarks(frame, results):

    """
    Draw landmarks on the frame for visualization.
    
    Args:
        frame (np.ndarray): The video frame on which to draw.
        results (mp - landmark_list): The MediaPipe results containing landmarks.
    
    Returns:
        np.ndarray: The video frame with visualization.
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    return frame

def predict_sign(video_path, model, label_encoder, confidence_threshold=0.4, visualization=True):
   
    """
    This function processes the video, extracts frame-level features using MediaPipe,
    makes a prediction with the provided model, and visualizes the prediction on the video.
    
    Args:
        video_path (str): Path to the video file.
        model (tf.keras.Model): ISL recognition model.
        label_encoder (LabelEncoder): Label encoder to decode model predictions.
        confidence_threshold (float): Minimum confidence to accept the prediction.
        visualization (bool): Whether to display the video with or without predicitons.
    
    Returns:
        tuple: A tuple containing:
            - final_prediction (str): Predicted class or "Uncertain" if below the confidence threshold.
            - confidence (float): The confidence level of the prediction.
    """
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    print("Collecting features from all frames...")
    frames_features = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        features, results = process_frame(frame, holistic)
        frames_features.append(features)
        
        current_frame += 1
        progress = (current_frame / total_frames) * 100
        print(f"\rProcessing frames: {progress:.1f}% complete", end="")
    
    print("\nFeature extraction completed.")
    
    if len(frames_features) > 0:
        print("\nMaking prediction...")
        X = np.array([frames_features])
        prediction = model.predict(X, verbose=0)
        confidence = np.max(prediction)
        predicted_idx = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        
        final_prediction = predicted_class if confidence >= confidence_threshold else "Uncertain"
        
        if visualization:
            print("\nStarting visualization with prediction...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to beginning
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                _, results = process_frame(frame, holistic)
                annotated_frame = draw_landmarks(frame.copy(), results)
                
                text = f"{final_prediction}: {confidence:.2%}"
                cv2.putText(annotated_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Sign Language Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
        cap.release()
        if visualization:
            cv2.destroyAllWindows()
        holistic.close()
        
        return final_prediction, confidence
    else:
        return "No features extracted", 0.0



def main():

    """
    Main function to load model and label encoder, process a video, and predict the sign language class.
    """

    model_path = "/path/to/model_file.keras" 
    encoder_path = "/path/to/label_encoder_file.keras"  
    video_path = "/path/to/test_video.mp4" 

    try:
        print("Loading model and label encoder...")
        model = load_model(model_path, safe_mode=False)
        label_encoder = load_label_encoder(encoder_path)
        
        print("Processing video...")
        predicted_class, confidence = predict_sign(
            video_path, 
            model, 
            label_encoder,
            confidence_threshold=0.2,
            visualization=True
        )
        
        print("\nPrediction Results:")
        print(f"Predicted Sign: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()

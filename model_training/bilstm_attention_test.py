import numpy as np
import pickle
import tensorflow as tf
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense
import keras


def process_frame_for_upper_body(frame):

    """
    Process a video frame to extract upper body and hand landmarks using MediaPipe.

    Args:
        frame (np.ndarray): A video frame in BGR format.

    Returns:
        list: A concatenated list of landmark coordinates (150 values) representing
              the upper body and both hands, or a list of 150 zeros in case of error.
    """

    mp_holistic = mp.solutions.holistic
    
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        upper_body_landmarks = np.zeros(24)    # 8 points * 3 coordinates
        left_hand_landmarks = np.zeros(63)     # 21 points * 3 coordinates
        right_hand_landmarks = np.zeros(63)    # 21 points * 3 coordinates
        
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
        
        features = np.concatenate([
            upper_body_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
        ]).tolist()
        
        return features
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return [0.0] * 150


def process_video(video_path):
    
    """
    Process a video file to extract features from each frame.

    Args:
        video_path (str): File path to the video.

    Returns:
        np.ndarray or None: An array of features extracted from the video frames,
                            or None if the video cannot be processed.
    """

    video_features = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_features = process_frame_for_upper_body(frame)
            if frame_features:
                video_features.append(frame_features)
                
        cap.release()
        
        if not video_features:
            print("No features extracted from video")
            return None
            
        return np.array(video_features)
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None


@keras.saving.register_keras_serializable()
class SelfAttention(Layer):

    """
    Implements a self-attention layer using scaled dot-product attention.

    Args:
        units (int): The number of units for the Dense projections.
    """

    def __init__(self, units, **kwargs):
        
        super().__init__(**kwargs)
        self.units = units
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None

    def build(self, input_shape):

        """
        Build the layer, initializing the Dense layers for query, key, and value.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """

        self.query_dense = Dense(self.units, use_bias=False)
        self.key_dense = Dense(self.units, use_bias=False)
        self.value_dense = Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):

        """
        Perform the forward pass using scaled dot-product attention.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, time, features).

        Returns:
            tf.Tensor: The output tensor after applying self-attention.
        """

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        scores = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        output = tf.matmul(attention_weights, value)
        return output

    def get_config(self):

        """
        Return the configuration of the layer for serialization.

        Returns:
            dict: A dictionary containing the layer configuration.
        """

        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config


def load_model_and_encoder(model_path, encoder_path):
    
    """
    Load the trained model and label encoder.

    Args:
        model_path (str): File path to the trained Keras ISL recognition model.
        encoder_path (str): File path to the label encoder.

    Returns:
        tuple: A tuple containing the loaded model and label encoder.
    """

    try:
        keras.config.enable_unsafe_deserialization()
        custom_objects = {
            'SelfAttention': SelfAttention
        }
        
        model = load_model(model_path, custom_objects=custom_objects)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        return model, label_encoder
    except Exception as e:
        print(f"Error loading model or encoder: {str(e)}")
        return None, None



def predict_sign(video_path, model_path, encoder_path):

    """
    Predict the sign language class from a video file.

    Args:
        video_path (str): File path to the video to be processed.
        model_path (str): File path to the pre-trained Keras model.
        encoder_path (str): File path to the label encoder.

    Returns:
        dict : A dictionary containing:
            - 'top_prediction' (str): The predicted class.
            - 'confidence' (float): Confidence level of the top prediction.
            - 'top_3_predictions' (list): A list of dictionaries for the top 3 predictions,
                                          each with 'sign' and 'confidence' keys.
    """

    model, label_encoder = load_model_and_encoder(model_path, encoder_path)
    if model is None or label_encoder is None:
        return None
    
    features = process_video(video_path)
    if features is None:
        return None
    
    features = np.expand_dims(features, axis=0)  
    
    try:
        predictions = model.predict(features)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        confidence = predictions[0][predicted_class_index]
        
        top_3_indices = np.argsort(predictions[0])[-10:][::-1]
        top_3_predictions = [
            {
                'sign': label_encoder.inverse_transform([idx])[0],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        return {
            'top_prediction': predicted_class,
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions
        }
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None


def main():

    """
    Main function to predict sign language from a video.
    """ 

    video_path = "/path/to/video.mp4"  
    model_path = "/path/to/model_file.keras" 
    encoder_path = "/path/to/label_encoder.pkl"  

    results = predict_sign(video_path, model_path, encoder_path)
    
    if results:
        print("\nPrediction Results:")
        print(f"Top Prediction: {results['top_prediction']}")
        print(f"Confidence: {results['confidence']*100:.2f}%")
        
        print("\nTop 3 Predictions:")
        for i, pred in enumerate(results['top_3_predictions'], 1):
            print(f"{i}. {pred['sign']} ({pred['confidence']*100:.2f}%)")
    else:
        print("Could not make prediction")

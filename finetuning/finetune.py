import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import mediapipe as mp
import cv2
import logging
import keras


mp_holistic = mp.solutions.holistic

keras.config.enable_unsafe_deserialization()

class SignLanguageDataGenerator(tf.keras.utils.Sequence):

    """
    Data generator for sign language sequences.

    This generator handles batching, optional data augmentation, and padding
    of variable-length sequences for training or fine-tuning a sign language model.

    Args:
        X (array): List of input sequences (each sequence is a NumPy array).
        y (array): Corresponding labels for the input sequences.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data after each epoch.
        augment (bool): Whether to apply data augmentation on the sequences.
    """

    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):  
        
        """
        Calculate the number of batches per epoch.

        Returns:
            int: Total number of batches per epoch.
        """

        return int(np.ceil(len(self.X) / self.batch_size))

    
    def augment_sequence(self, sequence):

        """
        Apply random data augmentation to a sequence.

        Args:
            sequence (np.ndarray): A NumPy array representing a sequence of feature vectors.

        Returns:
            np.ndarray: The augmented sequence.
        """

        augmented = sequence.copy()
        
        # Add random noise
        # if np.random.random() > 0.5:
        #     noise = np.random.normal(0, 0.01, sequence.shape)
        #     augmented = augmented + noise
        
        # Random temporal masking
        if np.random.random() > 0.5:
            mask_size = int(sequence.shape[0] * 0.1)
            start_idx = np.random.randint(0, sequence.shape[0] - mask_size)
            augmented[start_idx:start_idx + mask_size] = 0
            
        # Random feature masking
        if np.random.random() > 0.5:
            feature_mask = np.random.random(sequence.shape[1]) > 0.1
            augmented = augmented * feature_mask
            
        return augmented

    def __getitem__(self, idx):

        """
        Generate one batch of data.

        Args:
            idx (int): Index of the batch.

        Returns:
            tuple: A tuple containing the padded input sequences and the corresponding labels.
        """

        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = [self.X[i] for i in batch_indexes]
        
        if self.augment:
            batch_X = [self.augment_sequence(seq) for seq in batch_X]
        
        max_len = max(len(seq) for seq in batch_X)
        padded_X = np.zeros((len(batch_X), max_len, batch_X[0].shape[1]))
        for i, seq in enumerate(batch_X):
            padded_X[i, :len(seq)] = seq
            
        return padded_X, self.y[batch_indexes]

    def on_epoch_end(self):

        """
        Shuffle indexes after each epoch if shuffling is enabled.
        """

        if self.shuffle:
            np.random.shuffle(self.indexes)

            

def load_model_and_encoder(model_file,label_encoder_file):

    """
    Load the trained model and label encoder.
    Args:
        model_file (str): The trained model file.
        label_encoder_file (str): The label encoder file.

    Returns:
        tuple: A tuple containing the loaded model and label encoder.
    """

    model = tf.keras.models.load_model(model_file)
    
    # model = tf.keras.models.load_model("/Users/aswath/Desktop/ISL/review1/model_new_review2_new/epoch_models/model_epoch_085.keras")
    # with open('/Users/aswath/Desktop/ISL/review1/model_new_review2_new/label_encoder.pkl', 'rb') as f:
    #     label_encoder = pickle.load(f)
        
    with open(label_encoder_file, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder



def process_frame_for_upper_body(frame):

    """
    Process a single video frame to extract upper body and hand landmarks.

    Args:
        frame (np.ndarray): A single video frame in BGR color format.

    Returns:
        array: A list of concatenated landmark coordinates (150 values) or a list of 150 zeros on error.
    """

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
        logging.error(f"Error in process_frame_for_upper_body: {str(e)}")
        return [0.0] * 150  

def process_new_video(video_path):

    """
    Process a new video file and extract features from each frame.

    Args:
        video_path (str): The file path to the video.

    Returns:
        np.ndarray: An array containing the extracted features for each frame.
    """

    video_features = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        features = process_frame_for_upper_body(frame)
        if features:
            video_features.append(features)
    
    cap.release()
    return np.array(video_features)


def prepare_fine_tuning_data(new_videos_dir, label_encoder, mp_holistic):

    """
    Prepare the additional/new video data for fine-tuning the model.

    Args:
        new_videos_dir (str): Directory containing new video classes.
        label_encoder (LabelEncoder): The label encoder used for encoding class labels.
        mp_holistic (module): The MediaPipe holistic module instance.

    Returns:
        tuple: A tuple containing the list of extracted video features and one-hot encoded labels.
    """

    X_new = []
    y_new = []
    
    original_num_classes = len(label_encoder.classes_)
    print(f"Original number of classes: {original_num_classes}")
    
    for class_name in os.listdir(new_videos_dir):
        class_dir = os.path.join(new_videos_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing class: {class_name}")

        for video_file in os.listdir(class_dir):
            if video_file.endswith(('.mp4', '.MOV')):
                video_path = os.path.join(class_dir, video_file)
                features = process_new_video(video_path, mp_holistic)
                
                if len(features) > 0:
                    X_new.append(features)
                    y_new.append(class_name)
    
    y_new_encoded = label_encoder.transform(y_new)
    y_new_categorical = tf.keras.utils.to_categorical(y_new_encoded, num_classes=original_num_classes)
    
    print(f"Processed {len(X_new)} videos for fine-tuning")
    print(f"Output shape: {y_new_categorical.shape}")
    
    return X_new, y_new_categorical
    

def fine_tune_model(model, X_new, y_new, model_save_dir):
    """
    Fine-tune the pre-trained model using the additional/new video data.

    Args:
        model (tf.keras.Model): The pre-trained model to be fine-tuned.
        X_new (list): New video data features.
        y_new (np.ndarray): One-hot encoded labels for the new data.
        model_save_dir (str): Directory to save the fine-tuned model and checkpoints.

    Returns:
        tuple: A tuple containing the fine-tuned model and the training history.
    """
    
    fine_tune_generator = SignLanguageDataGenerator(
        X_new, y_new,
        batch_size=8,  
        shuffle=True,
        augment=True 
    )
    
    optimizer = Adam(learning_rate=0.0001)  
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        # EarlyStopping(
        #     monitor='loss',
        #     patience=5,
        #     restore_best_weights=True,
        #     verbose=1
        # ),
        ModelCheckpoint(
            os.path.join(model_save_dir, 'fine_tuned_model.keras'),
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        fine_tune_generator,
        epochs=100,  
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def main_fine_tuning():

    """
    Main function to fine-tune the pre-trained ISL model.
    """

    # model_dir = '/Users/aswath/Desktop/ISL/review1/model_new_review2_new'  
    # new_videos_dir = '/Users/aswath/Desktop/ISL/test'  
    # fine_tuned_save_dir = '/Users/aswath/Desktop/ISL/review1/fine_tune_model'

    new_videos_dir = '/path/to/new_videos_dir'  
    fine_tuned_save_dir = '/path/to/fine_tuned_model_dir'
    model_file = "/path/to/model_file.keras"
    label_encoder_file = "/path/to/label_encoder_file.pkl"
    
    os.makedirs(fine_tuned_save_dir, exist_ok=True)
    
    mp_holistic = mp.solutions.holistic
    
    print("Loading existing model and label encoder...")
    model, label_encoder = load_model_and_encoder(model_file, label_encoder_file)
    
    print("Processing new videos...")
    X_new, y_new = prepare_fine_tuning_data(new_videos_dir, label_encoder, mp_holistic)
    
    if len(X_new) == 0:
        print("No new data found! Please add videos to the new_examples directory.")
        return
    
    print(f"Found {len(X_new)} new examples for fine-tuning")
    
    print("Starting fine-tuning...")
    fine_tuned_model, history = fine_tune_model(model, X_new, y_new, fine_tuned_save_dir)
    
    print("Saving fine-tuned model and updated label encoder...")
    fine_tuned_model.save(os.path.join(fine_tuned_save_dir, 'final_fine_tuned_model.keras'))
    
    with open(os.path.join(fine_tuned_save_dir, 'updated_label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open(os.path.join(fine_tuned_save_dir, 'fine_tuning_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main_fine_tuning()
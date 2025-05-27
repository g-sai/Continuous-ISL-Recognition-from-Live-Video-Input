import os
import pickle
import json
import logging
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation, Add, Bidirectional, Concatenate,
    Conv1D, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D, GRU, Input,
    Lambda, LayerNormalization, Multiply, SpatialDropout1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2




mp_holistic = mp.solutions.holistic
results_queue = Queue()

    
def is_directory_processed(dir_path):

    """
    Check if a directory has already been processed for feature extraction.
    
    Args:
        dir_path (str): Path to the directory to check.
        
    Returns:
        bool: True if the directory exists and contains at least one file ending with '_features.pkl', else False.
    """

    if not os.path.exists(dir_path):
        return False
    
    pickle_files = [f for f in os.listdir(dir_path) if f.endswith('_features.pkl')]
    return len(pickle_files) > 0


def extract_features_threaded(base_dir, feature_vectors_dir, max_workers=2):

    """
    Extract features from videos in a directory using threaded processing.
    
    Args:
        base_dir (str): The root directory containing video subdirectories.
        feature_vectors_dir (str): Directory to store extracted feature pickle files.
        max_workers (int): Maximum number of worker threads to use.
        
    Returns:
        tuple: Two lists containing extracted features (X_data) and their corresponding labels (y_labels).
    """

    X_data = []
    y_labels = []
    video_paths = []
    
    total_dirs = 0
    processed_dirs = 0
    dirs_to_process = 0
    
    for root, dirs, files in os.walk(base_dir):
        relative_path = os.path.relpath(root, base_dir)
        if relative_path == '.':
            continue
            
        total_dirs += 1
        output_dir = os.path.join(feature_vectors_dir, relative_path)
        
        if is_directory_processed(output_dir):
            processed_dirs += 1
            print(f"Skipping already processed directory: {relative_path}")
            continue
            
        dirs_to_process += 1
        print(f"Will process directory: {relative_path}")
        
        for file in files:
            if file.endswith(('.MOV', '.mp4')):
                video_path = os.path.join(root, file)
                os.makedirs(output_dir, exist_ok=True)
                video_paths.append((video_path, output_dir))

    print(f"\nDirectory Processing Summary:")
    print(f"Total directories found: {total_dirs}")
    print(f"Already processed directories: {processed_dirs}")
    print(f"Directories to process: {dirs_to_process}")
    print(f"Total videos to process: {len(video_paths)}\n")

    if len(video_paths) == 0:
        print("No new videos to process!")
        return X_data, y_labels

    start_time = time.time()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for args in video_paths:
                future = executor.submit(process_video_threaded, args)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    while not results_queue.empty():
                        features, label = results_queue.get()
                        if features:  
                            X_data.append(features)
                            y_labels.append(label)
                except Exception as e:
                    print(f"Error processing results: {str(e)}")
                    continue

    except Exception as e:
        print(f"Error in thread pool execution: {str(e)}")

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(X_data)} videos")
    
    return X_data, y_labels


def process_frame_for_upper_body(frame):

    """
    Process a video frame to extract upper body and hand landmarks using MediaPipe.
    
    Args:
        frame (np.ndarray): A video frame in BGR format.
        
    Returns:
        list: A concatenated list of landmark coordinates (150 values) or a list of 150 zeros in case of error.
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

def process_video_threaded(args):

    """
    Process a single video file in a threaded context to extract features.
    
    Args:
        args (tuple): A tuple containing the video path and the directory to store feature files.
        
    Returns:
        list or None: A list of extracted features for the video or None if the video cannot be processed.
    """

    video_path, feature_vectors_dir = args
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    feature_file_path = os.path.join(feature_vectors_dir, f'{video_name}_features.pkl')
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
        
        if video_features:
            try:
                with open(feature_file_path, 'wb') as f:
                    pickle.dump(video_features, f)
                
                results_queue.put((video_features, os.path.basename(os.path.dirname(video_path))))
                print(f"Successfully processed {video_name}")
            except Exception as e:
                print(f"Error saving features for {video_name}: {str(e)}")
        else:
            print(f"No features extracted from {video_name}")
            
    except Exception as e:
        print(f"Error processing video {video_name}: {str(e)}")
    
    return video_features


def load_features_from_file(file_path):

    """
    Load feature data from a pickle file.
    
    Args:
        file_path (str): The path to the file containing feature data.
        
    Returns:
        The feature data.
    """

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_all_features(feature_vectors_dir):

    """
    Load all extracted features from a directory structure.
    
    Args:
        feature_vectors_dir (str): The root directory containing feature vector subdirectories.
        
    Returns:
        tuple: Two lists containing the feature arrays (X_data) and their corresponding labels (y_labels).
    """

    X_data = []
    y_labels = []

    sub_dirs = [d for d in os.listdir(feature_vectors_dir) 
                if os.path.isdir(os.path.join(feature_vectors_dir, d))]

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(feature_vectors_dir, sub_dir)
        feature_files = [f for f in os.listdir(sub_dir_path) 
                        if f.endswith('_features.pkl')]

        for feature_file in feature_files:
            file_path = os.path.join(sub_dir_path, feature_file)
            features = load_features_from_file(file_path)
            if features:
                features = np.array(features)
                X_data.append(features)
                y_labels.append(sub_dir)
    
    return X_data, y_labels


def prepare_data(X_data, y_labels):

    """
    Prepare feature data and labels for training or evaluation.
    
    Args:
        X_data (list): List of feature arrays.
        y_labels (list): List of labels corresponding to the feature arrays.
        
    Returns:
        tuple: A tuple containing:
            - A tuple with training, two validation splits, and corresponding labels.
            - The fitted LabelEncoder instance.
    """

    X_data = [np.array(seq) for seq in X_data]
    
    label_encoder = LabelEncoder()
    y_labels_encoded = label_encoder.fit_transform(y_labels)
    y_labels_categorical = to_categorical(y_labels_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_labels_categorical, test_size=0.2, random_state=42
    )
    
    X_val1, X_val2, y_val1, y_val2 = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    return (X_train, X_val1, X_val2, y_train, y_val1, y_val2), label_encoder



def build_model(feature_dim, num_classes):

    """
    Build a sign language recognition model.
    
    Args:
        feature_dim (int): The dimension of each feature vector.
        num_classes (int): The number of target classes.
        
    Returns:
        Model: The compiled Keras model.
    """

    inputs = Input(shape=(None, feature_dim))

    upper_body = Lambda(lambda x: x[:, :, :24], name='split_upper_body')(inputs)
    left_hand = Lambda(lambda x: x[:, :, 24:87], name='split_left_hand')(inputs)
    right_hand = Lambda(lambda x: x[:, :, 87:], name='split_right_hand')(inputs)

    def create_feature_processor(input_tensor, units, name_prefix):
        x = LayerNormalization(name=f"{name_prefix}_norm1")(input_tensor)
        
        # Multi-scale temporal convolutions
        conv1 = Conv1D(units//2, kernel_size=3, padding='same', name=f"{name_prefix}_conv1")(x)
        conv2 = Conv1D(units//2, kernel_size=5, padding='same', name=f"{name_prefix}_conv2")(x)
        
        x = Concatenate(name=f"{name_prefix}_concat")([conv1, conv2])
        x = LayerNormalization(name=f"{name_prefix}_norm2")(x)
        x = Activation('relu', name=f"{name_prefix}_relu")(x)
        x = SpatialDropout1D(0.1, name=f"{name_prefix}_dropout")(x)
        
        # Self-attention
        attention = Dense(units, activation='tanh', name=f"{name_prefix}_att1")(x)
        attention = Dense(1, activation='sigmoid', name=f"{name_prefix}_att2")(attention)
        x = Multiply(name=f"{name_prefix}_att_mul")([x, attention])
        
        return x

    upper_body_features = create_feature_processor(upper_body, 32, "upper")
    left_hand_features = create_feature_processor(left_hand, 48, "left")
    right_hand_features = create_feature_processor(right_hand, 48, "right")

    x = Concatenate(name="merge_features")([
        upper_body_features, 
        left_hand_features, 
        right_hand_features
    ])
    
    x = Conv1D(128, kernel_size=1, name="feature_transform")(x)
    x = LayerNormalization(name="transform_norm")(x)
    x = Activation('relu', name="transform_relu")(x)
    x = SpatialDropout1D(0.2, name="transform_dropout")(x)

    gru1 = Bidirectional(GRU(64, return_sequences=True, 
                            kernel_regularizer=l2(0.01),
                            recurrent_regularizer=l2(0.01)), 
                        name="gru1")(x)
    gru1 = LayerNormalization(name="gru1_norm")(gru1)
    gru1 = SpatialDropout1D(0.3, name="gru1_dropout")(gru1)
    
    gru1_projected = Conv1D(128, kernel_size=1, name="gru1_proj")(gru1)
    x = Add(name="residual1")([x, gru1_projected])
    
    gru2 = Bidirectional(GRU(64, return_sequences=True,
                            kernel_regularizer=l2(0.01),
                            recurrent_regularizer=l2(0.01)),
                        name="gru2")(x)
    gru2 = LayerNormalization(name="gru2_norm")(gru2)
    gru2 = SpatialDropout1D(0.3, name="gru2_dropout")(gru2)
    
    gru2_projected = Conv1D(128, kernel_size=1, name="gru2_proj")(gru2)
    x = Add(name="residual2")([x, gru2_projected])

    # Temporal attention
    attention = Dense(128, activation='tanh', name="temporal_att1")(x)
    attention = Dense(1, activation='sigmoid', name="temporal_att2")(attention)
    x = Multiply(name="temporal_att_mul")([x, attention])
    
    avg_pool = GlobalAveragePooling1D(name="avg_pool")(x)
    max_pool = GlobalMaxPooling1D(name="max_pool")(x)
    x = Concatenate(name="merge_pools")([avg_pool, max_pool])

    x = Dense(128, use_bias=False, name="dense1")(x)
    x = LayerNormalization(name="dense1_norm")(x)
    x = Activation('relu', name="dense1_relu")(x)
    x = Dropout(0.5, name="dense1_dropout")(x)

    x = Dense(64, use_bias=False, name="dense2")(x)
    x = LayerNormalization(name="dense2_norm")(x)
    x = Activation('relu', name="dense2_relu")(x)
    x = Dropout(0.5, name="dense2_dropout")(x)

    outputs = Dense(num_classes, activation='softmax', 
                   kernel_regularizer=l2(0.01),
                   name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model





def create_directory_structure(base_dir, output_dir):
        
    """
    Create a replicated directory structure in the output directory based on the base directory.
        
    Args:
        base_dir (str): The base directory to replicatw.
        output_dir (str): The directory where the replicated structure will be created.
    """

    for root, dirs, files in os.walk(base_dir):
        relative_path = os.path.relpath(root, base_dir)
        if relative_path != '.':
            os.makedirs(os.path.join(output_dir, relative_path), exist_ok=True)


class SignLanguageDataGenerator(tf.keras.utils.Sequence):

    """
    Data generator for sign language sequences.
    
    This generator handles batching, optional data augmentation, and padding
    of variable-length sequences for training a sign language model.
    
    Args:
        X (array): List of input sequences.
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
        Returns the number of batches per epoch.
        
        Returns:
            int: Total number of batches.
        """

        return int(np.ceil(len(self.X) / self.batch_size))

    
    def augment_sequence(self, sequence):

        """
        Apply random data augmentation to a sequence.
        
        This function applies noise, random temporal and feature masking.
        
        Args:
            sequence (np.ndarray): A sequence of feature vectors.
            
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
            idx (int): Batch index.
            
        Returns:
            tuple: A tuple containing padded input sequences and corresponding labels.
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
        Shuffle the indexes after each epoch if shuffling is needed.
        """

        if self.shuffle:
            np.random.shuffle(self.indexes)
            


def main():

    """
    Main function to orchestrate data preparation, model building, training, and evaluation.
    """

    base_dir = '/path/to/augmented_dataset'
    feature_vectors_dir = '/path/to/feature_vectors_dir'
    model_save_dir = '/path/to/model_dir'

    create_directory_structure(base_dir, feature_vectors_dir)

    # Extract features
    X_data, y_labels = extract_features_threaded(base_dir, feature_vectors_dir, max_workers=2)
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, 'epoch_models'), exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, 'final'), exist_ok=True)
    
    print("Loading features...")
    X_data, y_labels = load_all_features(feature_vectors_dir)
    
    print("Preparing data...")
    (X_train, X_val1, X_val2, y_train, y_val1, y_val2), label_encoder = prepare_data(X_data, y_labels)
    
    print("Saving label encoder...")
    with open(os.path.join(model_save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    feature_dim = X_data[0].shape[1]
    num_classes = len(label_encoder.classes_)
    print(f"Feature dimension: {feature_dim}, Number of classes: {num_classes}")
    
    print("Building model...")
    model = build_model(feature_dim, num_classes)
    
    initial_learning_rate = 0.0005
    
    print("Compiling model...")
    optimizer = Adam(learning_rate=initial_learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    class SaveModelPerEpoch(tf.keras.callbacks.Callback):
        def __init__(self, save_dir):
            super().__init__()
            self.save_dir = save_dir
        
        def on_epoch_end(self, epoch, logs=None):
            epoch_filename = os.path.join(
                self.save_dir, 
                'epoch_models', 
                f'model_epoch_{epoch+1:03d}.keras'
            )
            self.model.save(epoch_filename)
            print(f'\nSaved model for epoch {epoch+1} as {epoch_filename}')
            
            # Save training info
            info = {
                'epoch': epoch + 1,
                'loss': float(logs.get('loss')),
                'accuracy': float(logs.get('accuracy')),
                'val_loss': float(logs.get('val_loss')),
                'val_accuracy': float(logs.get('val_accuracy'))
            }
            with open(os.path.join(self.save_dir, 'training_info.json'), 'a') as f:
                json.dump(info, f)
                f.write('\n')
    
    print("Setting up callbacks...")
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_save_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        SaveModelPerEpoch(model_save_dir)
    ]
    

    print("Creating data generators...")
    batch_size = 8
    train_generator = SignLanguageDataGenerator(
        X_train, y_train, 
        batch_size=batch_size, 
        shuffle=True
    )
    validation_generator1 = SignLanguageDataGenerator(
        X_val1, y_val1, 
        batch_size=batch_size, 
        shuffle=False
    )
    validation_generator2 = SignLanguageDataGenerator(
        X_val2, y_val2, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator1,
        epochs=220,
        callbacks=callbacks,
        verbose=1
    )
    
    final_model_path = os.path.join(model_save_dir, 'final', 'final_model.keras')
    model.save(final_model_path)
    
    history_path = os.path.join(model_save_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    print("\nEvaluating final model...")
    test_loss, test_accuracy = model.evaluate(validation_generator2, verbose=1)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    
    evaluation_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'classes': label_encoder.classes_.tolist()
    }
    
    with open(os.path.join(model_save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print("\nTraining completed successfully!")


if __name__=='__main__':
    main()


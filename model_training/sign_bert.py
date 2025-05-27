import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
import pickle
from tqdm import tqdm
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Activation 
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Embedding, GlobalAveragePooling1D
from tqdm import tqdm


"""
This was initially developed, but didn't give satisfactory results.
The layers should be carefully worked upon to imporve the performance of this model"""


def extract_features(video_path, holistic):
    print(f"Processing video: {video_path}")  
    frames_features = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))
        
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                left_hand[i] = [landmark.x, landmark.y, landmark.z]
        
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                right_hand[i] = [landmark.x, landmark.y, landmark.z]
        
        combined_features = np.concatenate([left_hand.flatten(), right_hand.flatten()])
        frames_features.append(combined_features)
        frame_count += 1
        
    cap.release()
    print(f"Processed {frame_count} frames") 
    return np.array(frames_features)

def preprocess_dataset(base_dir, cache_dir='preprocessed_features'):
    print("Starting dataset preprocessing...")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    mp_holistic = mp.solutions.holistic
    processed_data = []
    labels = []
    
    video_paths = []
    video_labels = []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            for video_file in os.listdir(label_dir):
                if video_file.endswith(('.mp4', '.avi', '.MOV')):
                    video_paths.append(os.path.join(label_dir, video_file))
                    video_labels.append(label)
    
    print(f"Found {len(video_paths)} videos to process")
    
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=2) as holistic:
        for video_path, label in tqdm(zip(video_paths, video_labels), total=len(video_paths)):
            cache_file = os.path.join(
                cache_dir, 
                f"{label}_{os.path.basename(video_path)}.npy"
            )
            
            if os.path.exists(cache_file):
                features = np.load(cache_file)
                print(f"Loaded cached features for {video_path}")
            else:
                features = extract_features(video_path, holistic)
                np.save(cache_file, features)
                print(f"Extracted and cached features for {video_path}")
            
            if len(features) > 0: 
                processed_data.append(features)
                labels.append(label)
    
    print("Dataset preprocessing completed!")
    return processed_data, labels

class SignLanguageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, labels, batch_size=32, shuffle=True, label_encoder=None):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_encoder = label_encoder
        self.indexes = np.arange(len(self.features))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.features) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_features = [self.features[i] for i in batch_indexes]
        
        max_length = max(len(seq) for seq in batch_features)
        padded_features = [
            np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 
                  mode='constant', constant_values=0)
            for seq in batch_features
        ]
        
        X = np.array(padded_features)
        
        if self.label_encoder:
            y = np.array([self.label_encoder[self.labels[i]] for i in batch_indexes])
        else:
            y = np.array([self.labels[i] for i in batch_indexes])
        
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.features))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def augment_sequence(self, sequence):
        augmented = sequence.copy()
        
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            new_length = int(len(sequence) * scale)
            augmented = tf.image.resize(
                augmented[np.newaxis, :, :],
                (new_length, sequence.shape[1])
            )[0].numpy()
        
        if np.random.random() < 0.3:
            mask_size = int(len(augmented) * 0.1)
            start = np.random.randint(0, len(augmented) - mask_size)
            augmented[start:start + mask_size] = 0
        
        if np.random.random() < 0.4:
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented = augmented + noise
        
        return augmented


class SignBERTConfig:
    def __init__(self):
        self.hidden_size = 512  
        self.num_attention_heads = 12 
        self.intermediate_size = 1024 
        self.num_hidden_layers = 8  
        self.max_position_embeddings = 512
        self.mask_prob = 0.15
        self.dropout_rate = 0.3
        self.attention_dropout = 0.2


class SignBERT(tf.keras.Model):
    def __init__(self, config, num_classes):
        super(SignBERT, self).__init__()
        self.config = config
        self.num_classes = num_classes
        
        self.gesture_extractor = HandStateExtractor(config)
        self.temporal_embedding = Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.dropout = Dropout(0.2)  
        
        self.encoder_blocks = [
            TransformerBlock(config) 
            for _ in range(config.num_hidden_layers)
        ]
        
        self.classifier = tf.keras.Sequential([
            GlobalAveragePooling1D(),
            Dense(512, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.5),
            Dense(256, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

    def get_config(self):
        config = super(SignBERT, self).get_config()
        config.update({
            'config': {
                'hidden_size': self.config.hidden_size,
                'num_attention_heads': self.config.num_attention_heads,
                'intermediate_size': self.config.intermediate_size,
                'num_hidden_layers': self.config.num_hidden_layers,
                'max_position_embeddings': self.config.max_position_embeddings,
                'mask_prob': self.config.mask_prob
            },
            'num_classes': self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        bert_config = SignBERTConfig()
        bert_config.hidden_size = config['config']['hidden_size']
        bert_config.num_attention_heads = config['config']['num_attention_heads']
        bert_config.intermediate_size = config['config']['intermediate_size']
        bert_config.num_hidden_layers = config['config']['num_hidden_layers']
        bert_config.max_position_embeddings = config['config']['max_position_embeddings']
        bert_config.mask_prob = config['config']['mask_prob']
        
        return cls(bert_config, config['num_classes'])
        
    def get_temporal_positions(self, batch_size, seq_length):
        positions = tf.range(seq_length, dtype=tf.int32)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [batch_size, 1])
        return positions
        
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        gesture_embeddings = self.gesture_extractor(inputs)
        
        positions = self.get_temporal_positions(batch_size, seq_length)
        temporal_embeddings = self.temporal_embedding(positions)
        x = gesture_embeddings + temporal_embeddings
        x = self.dropout(x, training=training)
        
        for block in self.encoder_blocks:
            x = block(x, training=training)
        
        return self.classifier(x)

class HandStateExtractor(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(HandStateExtractor, self).__init__(**kwargs)
        self.config = config
        
        self.feature_extractor = tf.keras.Sequential([
            LayerNormalization(epsilon=1e-6),
            
            Dense(256, activation=None),
            LayerNormalization(epsilon=1e-6),
            Activation('relu'),
            Dropout(0.2),
            
            Dense(512, activation=None),
            LayerNormalization(epsilon=1e-6),
            Activation('relu'),
            Dropout(0.2),
            
            Dense(config.hidden_size)
        ])
        
        self.norm = LayerNormalization(epsilon=1e-6)
    
    def normalize_hands(self, inputs):
        shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [shape[0], shape[1], 2, 21, 3])
        
        centers = tf.reduce_mean(inputs, axis=3, keepdims=True)
        
        centered = inputs - centers
        scale = tf.sqrt(tf.reduce_sum(tf.square(centered), axis=[3, 4], keepdims=True))
        normalized = centered / (scale + 1e-6)
        
        return tf.reshape(normalized, [shape[0], shape[1], 126])
    
    def call(self, inputs, training=False):
        normalized = self.normalize_hands(inputs)
        x = self.feature_extractor(normalized, training=training)
        return self.norm(x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.config = config
        self.att = MultiHeadAttention(
            num_heads=config.num_attention_heads,
            key_dim=config.hidden_size // config.num_attention_heads
        )
        self.ffn = tf.keras.Sequential([
            Dense(config.intermediate_size, activation='relu'),
            Dense(config.hidden_size),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'config': {
                'hidden_size': self.config.hidden_size,
                'num_attention_heads': self.config.num_attention_heads,
                'intermediate_size': self.config.intermediate_size
            }
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        bert_config = SignBERTConfig()
        bert_config.hidden_size = config['config']['hidden_size']
        bert_config.num_attention_heads = config['config']['num_attention_heads']
        bert_config.intermediate_size = config['config']['intermediate_size']
        return cls(bert_config)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def train_signbert():
    base_dir = '/Users/aswath/Desktop/SL/dataset_augmented'
    cache_dir = 'preprocessed_features'
    batch_size = 16
    
    print("Starting data preprocessing...")
    processed_features, labels = preprocess_dataset(base_dir, cache_dir)
    print(f"Preprocessing complete. Found {len(processed_features)} valid samples")
    
    unique_labels = sorted(set(labels))
    label_encoder = {label: i for i, label in enumerate(unique_labels)}
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved with mappings:", label_encoder)
    
    indices = np.arange(len(processed_features))
    np.random.shuffle(indices)
    train_split = int(0.85 * len(indices))
    val_split = int(0.95 * len(indices))
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    train_generator = SignLanguageDataGenerator(
        [processed_features[i] for i in train_indices],
        [labels[i] for i in train_indices],
        batch_size=batch_size,
        label_encoder=label_encoder
    )
    
    val_generator = SignLanguageDataGenerator(
        [processed_features[i] for i in val_indices],
        [labels[i] for i in val_indices],
        batch_size=batch_size,
        shuffle=False,
        label_encoder=label_encoder
    )
    
    test_generator = SignLanguageDataGenerator(
        [processed_features[i] for i in test_indices],
        [labels[i] for i in test_indices],
        batch_size=batch_size,
        shuffle=False,
        label_encoder=label_encoder
    )
    
    config = SignBERTConfig()
    num_classes = len(unique_labels)
    model = SignBERT(config, num_classes)
    
    feature_dim = processed_features[0].shape[1]
    model.build(input_shape=(None, None, feature_dim))
    
    initial_learning_rate = 1e-4
    warmup_steps = 1000
    decay_steps = len(train_generator) * 50  
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        decay_steps,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint(
            'isl_signbert',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("Starting model training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=callbacks,
        class_weight=compute_class_weights(labels, label_encoder)
    )
    
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    return history, model, label_encoder

def compute_class_weights(labels, label_encoder):
    encoded_labels = [label_encoder[label] for label in labels]
    class_counts = np.bincount(encoded_labels)
    total = len(encoded_labels)
    class_weights = {i: total / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    return class_weights


def load_trained_model_and_labels(model_path='isl_signbert.keras', label_mapping_path='label_encoder.pkl'):
    model = tf.keras.models.load_model(model_path, 
                                     custom_objects={
                                         'SignBERT': SignBERT,
                                         'HandStateExtractor': HandStateExtractor,
                                         'TransformerBlock': TransformerBlock
                                     })
    
    try:
        with open(label_mapping_path, 'rb') as f:
            label_encoder = pickle.load(f)
            print("Loaded label encoder with classes:", list(label_encoder.keys()))
    except FileNotFoundError:
        print("Label encoder mapping not found. Please provide the mapping dictionary.")
        return model, None
        
    return model, label_encoder

def process_video_for_prediction(video_path, display_output=True, output_path=None):
    try:
        print("Loading model and label encoder...")
        model = tf.keras.models.load_model('isl_signbert_best.keras', 
                                         custom_objects={
                                             'SignBERT': SignBERT,
                                             'HandStateExtractor': HandStateExtractor,
                                             'TransformerBlock': TransformerBlock
                                         })
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            print("Available classes:", list(label_encoder.keys()))
            
        idx_to_label = {v: k for k, v in label_encoder.items()}
        
    except FileNotFoundError as e:
        print(f"Error loading model or label encoder: {e}")
        print("Please ensure both 'isl_signbert_best.keras' and 'label_encoder.pkl' exist")
        return None
    except Exception as e:
        print(f"Unexpected error while loading model: {e}")
        return None
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    all_features = []
    processed_frames = []
    
    print("Processing video frames...")
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=2) as holistic:
        for _ in tqdm(range(total_frames)):
            success, frame = cap.read()
            if not success:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            left_hand = np.zeros((21, 3))
            right_hand = np.zeros((21, 3))
            
            if results.left_hand_landmarks:
                for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                    left_hand[i] = [landmark.x, landmark.y, landmark.z]
                mp_drawing.draw_landmarks(
                    frame, 
                    results.left_hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )
            
            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    right_hand[i] = [landmark.x, landmark.y, landmark.z]
                mp_drawing.draw_landmarks(
                    frame, 
                    results.right_hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            combined_features = np.concatenate([left_hand.flatten(), right_hand.flatten()])
            all_features.append(combined_features)
            processed_frames.append(frame)
    
    cap.release()
    
    if len(all_features) == 0:
        print("No hand features could be extracted from the video")
        return None
    
    input_features = np.expand_dims(np.array(all_features), axis=0)
    
    print("Making prediction...")
    predictions = model.predict(input_features, verbose=0)
    
    top_3_indices = np.argsort(predictions[0])[-10:][::-1]
    top_3_predictions = []
    
    for idx in top_3_indices:
        label = idx_to_label[idx]
        confidence = float(predictions[0][idx])
        top_3_predictions.append((label, confidence))
    
    predicted_class_idx = top_3_indices[0]
    predicted_label = idx_to_label[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    if display_output or output_path:
        print("Processing output video...")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        for frame in tqdm(processed_frames):
            text_size = cv2.getTextSize(f"Predicted: {predicted_label}", font, font_scale, thickness)[0]
            cv2.rectangle(frame, 
                         (10-padding, 10-padding),
                         (max(text_size[0]+30, 300), 140),
                         (0, 0, 0),
                         -1)
            
            y_pos = 30
            cv2.putText(frame, 
                       f"Predicted: {predicted_label}", 
                       (10, y_pos), 
                       font, 
                       font_scale, 
                       (0, 255, 0), 
                       thickness)
            
            cv2.putText(frame, 
                       f"Confidence: {confidence:.2f}", 
                       (10, y_pos + 30), 
                       font, 
                       font_scale, 
                       (0, 255, 0), 
                       thickness)
            
            y_pos += 60
            for i, (label, conf) in enumerate(top_3_predictions):
                cv2.putText(frame,
                           f"{i+1}. {label}: {conf:.2f}",
                           (10, y_pos),
                           font,
                           font_scale,
                           (0, 255, 0),
                           thickness)
                y_pos += 20
            
            if display_output:
                cv2.imshow('Prediction Result', frame)
                if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                    break
            
            if output_path:
                out.write(frame)
    
    if output_path:
        out.release()
    
    if display_output:
        cv2.destroyAllWindows()
    
    return {
        'predicted_label': predicted_label,
        'confidence': confidence,
        'top_3_predictions': top_3_predictions,
        'num_frames_processed': len(all_features)
    }


def test_video(video_path, display=True, save_output=None):
    print(f"Processing video: {video_path}")
    results = process_video_for_prediction(
        video_path,
        display_output=display,
        output_path=save_output
    )
    
    if results:
        print("\nPrediction Results:")
        print(f"Predicted Label: {results['predicted_label']}")
        print(f"Confidence: {results['confidence']:.2f}")
        print("\nTop 3 Predictions:")
        for i, (label, conf) in enumerate(results['top_3_predictions']):
            print(f"{i+1}. {label}: {conf:.2f}")
        print(f"\nFrames Processed: {results['num_frames_processed']}")
    else:
        print("Prediction failed")

def main():
    train_signbert()
    test_video('/path/to/test/video.mp4')




if __name__ == "__main__":
    main()


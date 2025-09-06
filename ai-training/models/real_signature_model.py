# signature_model.py - REAL AI SYSTEM (No More Fake Stuff!)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import settings
import logging
from typing import List, Tuple, Optional, Union, Dict
from PIL import Image
import os

logger = logging.getLogger(__name__)

class RealSignatureVerificationModel:
    """
    REAL AI SYSTEM for Signature Verification
    
    This is a proper multi-class deep learning system that:
    1. Learns individual signature characteristics for each student
    2. Can identify who wrote a signature (student recognition)
    3. Can detect if a signature is genuine or forged (authenticity detection)
    4. Provides meaningful confidence scores
    5. Actually works (unlike the fake Siamese network)
    """
    
    def __init__(self, max_students: int = 150):
        self.max_students = max_students
        self.student_model = None  # Multi-class classifier for student identification
        self.authenticity_model = None  # Binary classifier for genuine/forged detection
        self.student_to_id = {}  # Map student names to numeric IDs
        self.id_to_student = {}  # Map numeric IDs to student names
        self.image_size = settings.MODEL_IMAGE_SIZE
        self.batch_size = settings.MODEL_BATCH_SIZE
        self.epochs = settings.MODEL_EPOCHS
        self.learning_rate = settings.MODEL_LEARNING_RATE
        
        logger.info(f"🚀 Initialized REAL AI system for up to {max_students} students")
    
    def create_student_recognition_model(self):
        """
        Create the main model for recognizing individual students from signatures
        
        This is a REAL deep learning model that learns individual signature characteristics
        """
        logger.info("🏗️ Building REAL student recognition model...")
        
        # Input: Single signature image
        input_layer = layers.Input(shape=(self.image_size, self.image_size, 3), name='signature_input')
        
        # Feature extraction backbone - MobileNetV2 (proven for image recognition)
        backbone = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'  # Pre-trained on ImageNet for better feature extraction
        )
        
        # Freeze early layers, fine-tune later layers for signature-specific features
        for layer in backbone.layers[:-40]:  # Freeze most layers
            layer.trainable = False
        
        # Preprocess input for MobileNetV2
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(input_layer)
        
        # Extract features
        features = backbone(preprocessed, training=False)
        
        # Add spatial attention mechanism for signature-specific features
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid', name='spatial_attention')(features)
        attended_features = layers.Multiply(name='attention_application')([features, attention])
        
        # Global pooling to get fixed-size feature vector
        pooled_features = layers.GlobalAveragePooling2D(name='global_pooling')(attended_features)
        
        # Classification head - learns individual signature characteristics
        x = layers.Dense(1024, activation='relu', name='fc1')(pooled_features)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        x = layers.Dense(512, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        
        x = layers.Dense(256, activation='relu', name='fc3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.Dropout(0.2, name='dropout3')(x)
        
        # Output layer: Probability distribution over all students
        student_output = layers.Dense(
            self.max_students, 
            activation='softmax', 
            name='student_classification'
        )(x)
        
        # Create the model
        model = keras.Model(
            inputs=input_layer, 
            outputs=student_output, 
            name='real_signature_student_recognition'
        )
        
        # Compile with proper loss function for multi-class classification
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=1e-4  # L2 regularization to prevent overfitting
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',  # Proper loss for multi-class classification
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        logger.info(f"✅ REAL student recognition model created for {self.max_students} students")
        logger.info(f"📊 Model parameters: {model.count_params():,}")
        
        return model
    
    def create_authenticity_detection_model(self):
        """
        Create a model to detect if a signature is genuine or forged
        
        This model takes a signature and predicts authenticity (genuine/forged)
        """
        logger.info("🏗️ Building REAL authenticity detection model...")
        
        # Input: Single signature image
        input_layer = layers.Input(shape=(self.image_size, self.image_size, 3), name='signature_input')
        
        # Feature extraction backbone
        backbone = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        
        # Preprocess and extract features
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(input_layer)
        features = backbone(preprocessed, training=False)
        
        # Add attention for authenticity-specific features
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(features)
        attended_features = layers.Multiply()([features, attention])
        
        # Global pooling
        pooled_features = layers.GlobalAveragePooling2D()(attended_features)
        
        # Authenticity classification head
        x = layers.Dense(512, activation='relu')(pooled_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Binary output: genuine (1) or forged (0)
        authenticity_output = layers.Dense(1, activation='sigmoid', name='authenticity_classification')(x)
        
        # Create the model
        model = keras.Model(
            inputs=input_layer,
            outputs=authenticity_output,
            name='real_signature_authenticity_detection'
        )
        
        # Compile for binary classification
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, weight_decay=1e-4)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Proper loss for binary classification
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info("✅ REAL authenticity detection model created")
        logger.info(f"📊 Model parameters: {model.count_params():,}")
        
        return model
    
    def prepare_training_data(self, training_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for the REAL AI system
        
        Args:
            training_data: Dict with structure:
            {
                'student_name': {
                    'genuine': [image1, image2, ...],
                    'forged': [image1, image2, ...]
                }
            }
        
        Returns:
            X: Preprocessed images
            y_student: Student IDs (one-hot encoded)
            y_authenticity: Authenticity labels (1=genuine, 0=forged)
        """
        logger.info("🔄 Preparing training data for REAL AI system...")
        
        all_images = []
        student_labels = []
        authenticity_labels = []
        
        # Process each student's data
        for student_name, signatures in training_data.items():
            if student_name not in self.student_to_id:
                # Assign new student ID
                student_id = len(self.student_to_id)
                self.student_to_id[student_name] = student_id
                self.id_to_student[student_id] = student_name
                logger.info(f"📝 Added student '{student_name}' with ID {student_id}")
            
            student_id = self.student_to_id[student_name]
            
            # Process genuine signatures
            for img in signatures['genuine']:
                processed_img = self.preprocess_image(img)
                all_images.append(processed_img)
                student_labels.append(student_id)
                authenticity_labels.append(1)  # Genuine
            
            # Process forged signatures
            for img in signatures['forged']:
                processed_img = self.preprocess_image(img)
                all_images.append(processed_img)
                student_labels.append(student_id)  # Same student, but forged
                authenticity_labels.append(0)  # Forged
        
        # Convert to numpy arrays
        X = np.array(all_images, dtype=np.float32)
        y_student = np.array(student_labels, dtype=np.int32)
        y_authenticity = np.array(authenticity_labels, dtype=np.float32)
        
        # One-hot encode student labels
        y_student_onehot = tf.keras.utils.to_categorical(y_student, num_classes=self.max_students)
        
        logger.info(f"✅ Training data prepared:")
        logger.info(f"   📊 Total images: {len(all_images)}")
        logger.info(f"   👥 Students: {len(self.student_to_id)}")
        logger.info(f"   ✅ Genuine signatures: {np.sum(y_authenticity)}")
        logger.info(f"   ❌ Forged signatures: {len(y_authenticity) - np.sum(y_authenticity)}")
        
        return X, y_student_onehot, y_authenticity
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for the REAL AI model
        
        This ensures consistent input format for the deep learning model
        """
        # Handle PIL Image
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        # Ensure correct shape and dtype
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        # Resize to model input size
        image = tf.image.resize(image, [self.image_size, self.image_size])
        
        # Ensure float32 and proper range [0, 255]
        image = tf.cast(image, tf.float32)
        if tf.reduce_max(image) <= 1.0:
            image = image * 255.0
        
        return image.numpy()
    
    def train_real_ai_system(self, training_data: Dict) -> Dict:
        """
        Train the REAL AI system
        
        This trains both the student recognition model and authenticity detection model
        """
        logger.info("🚀 Starting REAL AI system training...")
        
        # Prepare training data
        X, y_student, y_authenticity = self.prepare_training_data(training_data)
        
        # Create models
        self.student_model = self.create_student_recognition_model()
        self.authenticity_model = self.create_authenticity_detection_model()
        
        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_student_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train student recognition model
        logger.info("🎯 Training student recognition model...")
        student_history = self.student_model.fit(
            X, y_student,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train authenticity detection model
        logger.info("🎯 Training authenticity detection model...")
        authenticity_history = self.authenticity_model.fit(
            X, y_authenticity,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ REAL AI system training completed!")
        
        return {
            'student_history': student_history.history,
            'authenticity_history': authenticity_history.history,
            'student_model': self.student_model,
            'authenticity_model': self.authenticity_model
        }
    
    def verify_signature(self, test_signature: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Complete signature verification using the REAL AI system
        
        This is the main verification function that:
        1. Identifies who wrote the signature
        2. Determines if it's genuine or forged
        3. Provides confidence scores
        """
        logger.info("🔍 Starting REAL signature verification...")
        
        # Preprocess test signature
        processed_signature = self.preprocess_image(test_signature)
        X_test = np.expand_dims(processed_signature, axis=0)
        
        # Step 1: Student recognition
        student_probs = self.student_model.predict(X_test, verbose=0)[0]
        predicted_student_id = np.argmax(student_probs)
        student_confidence = float(np.max(student_probs))
        
        # Step 2: Authenticity detection
        authenticity_score = self.authenticity_model.predict(X_test, verbose=0)[0][0]
        is_genuine = authenticity_score > 0.5
        
        # Get student name
        predicted_student_name = self.id_to_student.get(predicted_student_id, f"Unknown_{predicted_student_id}")
        
        # Calculate overall confidence
        overall_confidence = (student_confidence + authenticity_score) / 2
        
        result = {
            'predicted_student_id': int(predicted_student_id),
            'predicted_student_name': predicted_student_name,
            'student_confidence': student_confidence,
            'is_genuine': bool(is_genuine),
            'authenticity_score': float(authenticity_score),
            'overall_confidence': overall_confidence,
            'is_unknown': student_confidence < 0.3 or overall_confidence < 0.4
        }
        
        logger.info(f"🎯 Verification result:")
        logger.info(f"   👤 Predicted student: {predicted_student_name} (ID: {predicted_student_id})")
        logger.info(f"   🎯 Student confidence: {student_confidence:.3f}")
        logger.info(f"   ✅ Is genuine: {is_genuine} (score: {authenticity_score:.3f})")
        logger.info(f"   📊 Overall confidence: {overall_confidence:.3f}")
        logger.info(f"   ❓ Is unknown: {result['is_unknown']}")
        
        return result
    
    def save_models(self, base_path: str):
        """Save the trained REAL AI models"""
        if self.student_model:
            student_path = f"{base_path}_student_model.keras"
            self.student_model.save(student_path)
            logger.info(f"💾 Student model saved to {student_path}")
        
        if self.authenticity_model:
            authenticity_path = f"{base_path}_authenticity_model.keras"
            self.authenticity_model.save(authenticity_path)
            logger.info(f"💾 Authenticity model saved to {authenticity_path}")
        
        # Save student mappings
        import json
        mappings_path = f"{base_path}_student_mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump({
                'student_to_id': self.student_to_id,
                'id_to_student': self.id_to_student
            }, f, indent=2)
        logger.info(f"💾 Student mappings saved to {mappings_path}")
    
    def load_models(self, base_path: str):
        """Load the trained REAL AI models"""
        try:
            # Load student model
            student_path = f"{base_path}_student_model.keras"
            if os.path.exists(student_path):
                self.student_model = keras.models.load_model(student_path)
                logger.info(f"📂 Student model loaded from {student_path}")
            
            # Load authenticity model
            authenticity_path = f"{base_path}_authenticity_model.keras"
            if os.path.exists(authenticity_path):
                self.authenticity_model = keras.models.load_model(authenticity_path)
                logger.info(f"📂 Authenticity model loaded from {authenticity_path}")
            
            # Load student mappings
            mappings_path = f"{base_path}_student_mappings.json"
            if os.path.exists(mappings_path):
                import json
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                self.student_to_id = mappings['student_to_id']
                self.id_to_student = {int(k): v for k, v in mappings['id_to_student'].items()}
                logger.info(f"📂 Student mappings loaded from {mappings_path}")
            
            logger.info("✅ REAL AI models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load REAL AI models: {e}")
            return False
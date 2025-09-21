"""
Teachable Signature Model - Global Classification Approach
Replicates Teachable Machine behavior for signature verification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import os
import json
import pickle

logger = logging.getLogger(__name__)

class TeachableSignatureModel:
    """
    Global signature classification model that behaves like Teachable Machine:
    - One global model for all students
    - Few-shot learning capability
    - Incremental learning (add new students)
    - Owner detection only (no forgery detection)
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 embedding_dim: int = 512,
                 learning_rate: float = 0.001,
                 max_students: int = 1000):
        
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.max_students = max_students
        
        # Global model components
        self.feature_extractor = None
        self.classifier = None
        self.global_model = None
        
        # Student management (like Teachable Machine classes)
        self.student_to_class = {}  # student_id -> class_index
        self.class_to_student = {}  # class_index -> student_id
        self.student_names = {}     # student_id -> student_name
        self.num_classes = 0
        
        # Training history
        self.training_history = {}
        
    def create_feature_extractor(self) -> keras.Model:
        """
        Create the feature extraction backbone (like Teachable Machine)
        Uses MobileNetV2 for efficient feature extraction
        """
        inputs = keras.Input(shape=(self.image_size, self.image_size, 3))
        
        # Use MobileNetV2 as backbone (pre-trained on ImageNet)
        base_model = keras.applications.MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Feature extraction pipeline
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(self.embedding_dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output features
        features = layers.Dense(self.embedding_dim, activation='linear', name='features')(x)
        
        self.feature_extractor = keras.Model(inputs, features, name='feature_extractor')
        return self.feature_extractor
    
    def create_classifier(self, num_classes: int) -> keras.Model:
        """
        Create the classification head (like Teachable Machine)
        This is retrained when new students are added
        """
        inputs = keras.Input(shape=(self.embedding_dim,), name='features')
        
        # Classification layers
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer (one class per student)
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        self.classifier = keras.Model(inputs, outputs, name='classifier')
        return self.classifier
    
    def create_global_model(self) -> keras.Model:
        """
        Create the complete global model (feature extractor + classifier)
        """
        if self.feature_extractor is None:
            self.create_feature_extractor()
        
        if self.classifier is None:
            self.create_classifier(self.num_classes)
        
        # Combine feature extractor and classifier
        inputs = keras.Input(shape=(self.image_size, self.image_size, 3))
        features = self.feature_extractor(inputs)
        predictions = self.classifier(features)
        
        self.global_model = keras.Model(inputs, predictions, name='global_signature_model')
        return self.global_model
    
    def add_student(self, student_id: str, student_name: str) -> int:
        """
        Add a new student to the global model (like adding a class in Teachable Machine)
        Returns the class index for this student
        """
        if student_id in self.student_to_class:
            return self.student_to_class[student_id]
        
        # Add new student
        class_index = self.num_classes
        self.student_to_class[student_id] = class_index
        self.class_to_student[class_index] = student_id
        self.student_names[student_id] = student_name
        self.num_classes += 1
        
        logger.info(f"Added student {student_id} ({student_name}) as class {class_index}")
        return class_index
    
    def prepare_training_data(self, training_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for global model
        training_data format: {student_id: [image_arrays]}
        """
        all_images = []
        all_labels = []
        
        for student_id, images in training_data.items():
            if not images:
                continue
                
            # Add student if not exists
            if student_id not in self.student_to_class:
                self.add_student(student_id, f"Student_{student_id}")
            
            class_index = self.student_to_class[student_id]
            
            # Add images and labels
            for image in images:
                all_images.append(image)
                all_labels.append(class_index)
        
        X = np.array(all_images)
        y = tf.keras.utils.to_categorical(all_labels, num_classes=self.num_classes)
        
        logger.info(f"Prepared {len(all_images)} images across {self.num_classes} classes")
        return X, y
    
    def train_global_model(self, training_data: Dict, epochs: int = 50, validation_split: float = 0.2) -> Dict:
        """
        Train the global model on all student data
        """
        logger.info("Starting global model training...")
        
        # Prepare training data
        X, y = self.prepare_training_data(training_data)
        
        if len(X) == 0:
            raise ValueError("No training data provided")
        
        # Create/update global model
        self.create_global_model()
        
        # Compile model
        self.global_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                'best_global_model.keras',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = self.global_model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        logger.info("Global model training completed")
        
        return self.training_history
    
    def incremental_train(self, new_student_data: Dict, epochs: int = 20) -> Dict:
        """
        Incrementally train the model with new student data
        This is like adding a new class in Teachable Machine
        """
        logger.info("Starting incremental training...")
        
        # Add new students
        for student_id, images in new_student_data.items():
            if student_id not in self.student_to_class:
                self.add_student(student_id, f"Student_{student_id}")
        
        # Prepare new training data
        X_new, y_new = self.prepare_training_data(new_student_data)
        
        if len(X_new) == 0:
            logger.warning("No new training data provided")
            return {}
        
        # Update classifier for new number of classes
        if self.classifier is None or self.classifier.output_shape[-1] != self.num_classes:
            self.create_classifier(self.num_classes)
            self.create_global_model()
        
        # Compile model
        self.global_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate * 0.1),  # Lower LR for fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Fine-tune on new data
        history = self.global_model.fit(
            X_new, y_new,
            epochs=epochs,
            validation_split=0.2,
            batch_size=16,
            verbose=1
        )
        
        logger.info("Incremental training completed")
        return history.history
    
    def predict_student(self, image: np.ndarray, confidence_threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        Predict which student owns the signature
        Returns (student_id, confidence) or (None, 0.0) if not recognized
        """
        if self.global_model is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        # Get predictions
        predictions = self.global_model.predict(image, verbose=0)
        max_confidence = np.max(predictions[0])
        predicted_class = np.argmax(predictions[0])
        
        # Check confidence threshold
        if max_confidence < confidence_threshold:
            return None, max_confidence
        
        # Get student ID
        student_id = self.class_to_student.get(predicted_class)
        return student_id, max_confidence
    
    def get_student_name(self, student_id: str) -> str:
        """Get student name by ID"""
        return self.student_names.get(student_id, f"Student_{student_id}")
    
    def save_model(self, filepath: str):
        """Save the global model and metadata"""
        if self.global_model is None:
            raise ValueError("Model not trained yet")
        
        # Save model
        self.global_model.save(f"{filepath}.keras")
        
        # Save metadata
        metadata = {
            'student_to_class': self.student_to_class,
            'class_to_student': self.class_to_student,
            'student_names': self.student_names,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'embedding_dim': self.embedding_dim
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the global model and metadata"""
        # Load model
        self.global_model = keras.models.load_model(f"{filepath}.keras")
        
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.student_to_class = metadata['student_to_class']
        self.class_to_student = metadata['class_to_student']
        self.student_names = metadata['student_names']
        self.num_classes = metadata['num_classes']
        self.image_size = metadata['image_size']
        self.embedding_dim = metadata['embedding_dim']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict:
        """Get model summary information"""
        if self.global_model is None:
            return {"error": "Model not trained yet"}
        
        return {
            "num_students": self.num_classes,
            "students": list(self.student_names.keys()),
            "model_parameters": self.global_model.count_params(),
            "training_history": self.training_history
        }
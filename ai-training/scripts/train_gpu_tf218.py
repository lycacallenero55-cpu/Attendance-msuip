#!/usr/bin/env python3
"""
GPU Training Script - TensorFlow 2.18 Compatible
Optimized for AWS Deep Learning AMI with GPU support
"""

import sys
import os
import json
import boto3
import numpy as np
from PIL import Image
import io
import traceback
import tensorflow as tf
from tensorflow import keras
import tempfile
import shutil
import base64

# Configure TensorFlow for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), using GPU acceleration")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, using CPU")

class SignaturePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.processed_count = 0
        self.error_count = 0
    
    def preprocess_signature(self, img_data, debug_name="unknown"):
        try:
            img = None
            if isinstance(img_data, str):
                try:
                    if img_data.startswith('data:'):
                        img_data = img_data.split(',')[1]
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    print(f"  Successfully loaded base64 image for {debug_name}")
                except Exception as e:
                    print(f"  Failed to decode base64 for {debug_name}: {e}")
                    return None
            elif isinstance(img_data, list):
                img_array = np.array(img_data, dtype=np.float32)
                print(f"  Processing array data for {debug_name}, shape: {img_array.shape}")
                
                # Handle different array shapes
                if len(img_array.shape) == 1:
                    total_pixels = len(img_array)
                    side = int(np.sqrt(total_pixels))
                    if side * side == total_pixels:
                        img_array = img_array.reshape(side, side)
                        print(f"    Reshaped flat array to {img_array.shape}")
                    else:
                        # Try common image sizes
                        common_sizes = [(224,224),(256,256),(128,128),(64,64)]
                        reshaped = False
                        for h,w in common_sizes:
                            if h*w == total_pixels:
                                img_array = img_array.reshape(h,w)
                                print(f"    Reshaped to common size: {img_array.shape}")
                                reshaped = True
                                break
                        if not reshaped:
                            for h,w in common_sizes:
                                if h*w*3 == total_pixels:
                                    img_array = img_array.reshape(h,w,3)
                                    print(f"    Reshaped to 3-channel: {img_array.shape}")
                                    reshaped = True
                                    break
                        if not reshaped:
                            print(f"    Cannot reshape array of size {total_pixels} to known image format")
                            return None
                elif len(img_array.shape) == 3:
                    if img_array.shape[0] in [1,3,4]:
                        img_array = np.transpose(img_array, (1,2,0))
                        print(f"    Transposed from CHW to HWC: {img_array.shape}")
                    elif img_array.shape[2] not in [1,3,4]:
                        print(f"    Unexpected shape: {img_array.shape}")
                        return None
                
                # Normalize pixel values
                if img_array.max() <= 1.0 and img_array.min() >= 0.0:
                    img_array = (img_array * 255).astype(np.uint8)
                    print("    Scaled normalized values to 0-255 range")
                elif img_array.max() > 255 or img_array.min() < 0:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    print("    Clamped values to 0-255 range")
                else:
                    img_array = img_array.astype(np.uint8)
                
                # Convert to RGB if needed
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                    print(f"    Converted grayscale to RGB: {img_array.shape}")
                elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                    print(f"    Converted single channel to RGB: {img_array.shape}")
                
                try:
                    img = Image.fromarray(img_array)
                    print(f"    Created PIL image from array: {img.size}")
                except Exception as e:
                    print(f"    Failed to create PIL image: {e}")
                    return None
            elif hasattr(img_data, 'size'):
                img = img_data
                print(f"  Using existing PIL image for {debug_name}: {img.size}")
            else:
                print(f"  Unknown image data type for {debug_name}: {type(img_data)}")
                return None
            
            if img is None:
                return None
            
            # Convert to RGB and resize
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("    Converted to RGB mode")
            
            original_size = img.size
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            print(f"    Resized from {original_size} to {img.size}")
            
            # Normalize to 0-1 range
            img_array = np.array(img, dtype=np.float32) / 255.0
            self.processed_count += 1
            return img_array
            
        except Exception as e:
            print(f"  Error processing {debug_name}: {str(e)}")
            print(f"  Stack trace: {traceback.format_exc()}")
            self.error_count += 1
            return None

class SignatureEmbeddingModel:
    def __init__(self, max_students=150):
        self.max_students = max_students
        self.embedding_dim = 128
        self.student_to_id = {}
        self.id_to_student = {}
        self.embedding_model = None
        self.classification_head = None
        self.siamese_model = None
    
    def train_models(self, training_data, epochs=25, validation_split=0.2):
        print("Starting model training...")
        all_images = []
        all_labels = []
        
        print(f"Processing {len(training_data)} students...")
        for idx, (student_name, data) in enumerate(training_data.items()):
            self.student_to_id[student_name] = idx
            self.id_to_student[idx] = student_name
            genuine_count = len(data.get('genuine', []))
            forged_count = len(data.get('forged', []))
            print(f"Student {student_name} (ID: {idx}): {genuine_count} genuine, {forged_count} forged")
            
            # Process genuine images
            for i, img in enumerate(data.get('genuine', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(idx)
                else:
                    print(f"    Skipping None genuine image {i} for {student_name}")
            
            # Process forged images
            for i, img in enumerate(data.get('forged', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(idx)
                else:
                    print(f"    Skipping None forged image {i} for {student_name}")
        
        print(f"Total images for training: {len(all_images)}")
        print(f"Unique student IDs: {len(set(all_labels))}")
        
        if len(all_images) == 0:
            raise ValueError("No valid training samples found after processing")
        
        if len(all_images) < 5:
            print("WARNING: Very few samples for training. Results may be poor.")
            validation_split = 0.0
        
        print("Converting to numpy arrays...")
        X = np.array(all_images)
        y = keras.utils.to_categorical(all_labels, num_classes=len(training_data))
        
        print(f"Final training data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Data type: {X.dtype}, range: [{float(X.min()):.3f}, {float(X.max()):.3f}]")
        
        # Build CNN model
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(224, 224, 3)),
            keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(training_data), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model summary:")
        model.summary()
        
        print(f"Starting training with validation_split={validation_split}")
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy' if validation_split > 0 else 'accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_split > 0 else 'loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            ),
        ]
        
        try:
            if validation_split > 0:
                history = model.fit(
                    X, y,
                    batch_size=min(32, len(X)),
                    epochs=epochs,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = model.fit(
                    X, y,
                    batch_size=min(32, len(X)),
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1
                )
        except Exception as e:
            print(f"Training failed: {e}")
            traceback.print_exc()
            raise
        
        self.classification_head = model
        self.embedding_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[-3].output
        )
        
        print("Training completed successfully!")
        return {'classification_history': history.history, 'siamese_history': {}}
    
    def save_models(self, base_path):
        """Save models to disk"""
        if self.classification_head:
            self.classification_head.save(f"{base_path}_classification.keras")
            print(f"Saved classification model to {base_path}_classification.keras")
        
        if self.embedding_model:
            self.embedding_model.save(f"{base_path}_embedding.keras")
            print(f"Saved embedding model to {base_path}_embedding.keras")
        
        # Save mappings
        mappings = {
            'student_to_id': self.student_to_id,
            'id_to_student': self.id_to_student
        }
        with open(f"{base_path}_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"Saved mappings to {base_path}_mappings.json")

def train_on_gpu(training_data_key, job_id, student_id):
    try:
        s3 = boto3.client('s3')
        bucket = os.environ.get('S3_BUCKET', 'signatureai-uploads')
        
        print(f"Starting training for job {job_id}")
        print(f"S3 bucket: {bucket}")
        print(f"Training data key: {training_data_key}")
        print(f"Downloading training data from s3://{bucket}/{training_data_key}")
        
        try:
            response = s3.get_object(Bucket=bucket, Key=training_data_key)
            training_data_raw = json.loads(response['Body'].read())
            print("Successfully downloaded training data")
        except Exception as e:
            print(f"Failed to download training data: {e}")
            traceback.print_exc()
            raise
        
        print(f"Raw training data contains {len(training_data_raw)} students")
        
        preprocessor = SignaturePreprocessor(target_size=(224, 224))
        model_manager = SignatureEmbeddingModel(max_students=150)
        
        print("Processing training data...")
        processed_data = {}
        
        for student_name, data in training_data_raw.items():
            print(f"\nProcessing student: {student_name}")
            genuine_images = []
            forged_images = []
            
            genuine_raw = data.get('genuine', [])
            print(f"  Found {len(genuine_raw)} genuine images")
            for i, img_data in enumerate(genuine_raw):
                print(f"    Processing genuine image {i+1}/{len(genuine_raw)}")
                processed_img = preprocessor.preprocess_signature(
                    img_data, 
                    debug_name=f"{student_name}_genuine_{i}"
                )
                if processed_img is not None:
                    genuine_images.append(processed_img)
                    print("      ✓ Successfully processed")
                else:
                    print("      ✗ Failed to process")
            
            forged_raw = data.get('forged', [])
            print(f"  Found {len(forged_raw)} forged images")
            for i, img_data in enumerate(forged_raw):
                print(f"    Processing forged image {i+1}/{len(forged_raw)}")
                processed_img = preprocessor.preprocess_signature(
                    img_data, 
                    debug_name=f"{student_name}_forged_{i}"
                )
                if processed_img is not None:
                    forged_images.append(processed_img)
                    print("      ✓ Successfully processed")
                else:
                    print("      ✗ Failed to process")
            
            processed_data[student_name] = {
                'genuine': genuine_images, 
                'forged': forged_images
            }
            
            total_processed = len(genuine_images) + len(forged_images)
            total_raw = len(genuine_raw) + len(forged_raw)
            success_rate = (total_processed / total_raw * 100) if total_raw > 0 else 0
            
            print(f"  Final: {len(genuine_images)} genuine, {len(forged_images)} forged")
            print(f"  Success rate: {total_processed}/{total_raw} ({success_rate:.1f}%)")
        
        print("\n=== PREPROCESSING SUMMARY ===")
        print(f"Total images processed successfully: {preprocessor.processed_count}")
        print(f"Total processing errors: {preprocessor.error_count}")
        
        total_samples = sum(len(d['genuine']) + len(d['forged']) for d in processed_data.values())
        print(f"Total processed samples available for training: {total_samples}")
        
        if total_samples == 0:
            raise ValueError("No valid training samples found after processing")
        
        validation_split = 0.0 if total_samples < 5 else 0.2
        
        print("\n=== STARTING MODEL TRAINING ===")
        training_result = model_manager.train_models(
            processed_data, 
            epochs=25, 
            validation_split=validation_split
        )
        
        print("Training completed! Saving models...")
        temp_dir = f'/tmp/{job_id}_models'
        os.makedirs(temp_dir, exist_ok=True)
        model_manager.save_models(f'{temp_dir}/signature_model')
        
        # Upload models to S3
        model_files = ['embedding', 'classification']
        model_urls = {}
        
        for model_type in model_files:
            file_path = f'{temp_dir}/signature_model_{model_type}.keras'
            if os.path.exists(file_path):
                s3_key = f'models/{job_id}/{model_type}.keras'
                s3.upload_file(file_path, bucket, s3_key)
                model_urls[model_type] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
                print(f"Uploaded {model_type} model to S3: {s3_key}")
            else:
                print(f"WARNING: {model_type} model file not found: {file_path}")
        
        # Upload mappings
        mappings_path = f'{temp_dir}/signature_model_mappings.json'
        if os.path.exists(mappings_path):
            s3_key = f'models/{job_id}/mappings.json'
            s3.upload_file(mappings_path, bucket, s3_key)
            model_urls['mappings'] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
            print(f"Uploaded mappings to S3: {s3_key}")
        
        # Extract accuracy metrics
        classification_history = training_result.get('classification_history', {})
        final_accuracy = None
        if 'accuracy' in classification_history:
            accuracies = classification_history['accuracy']
            if accuracies:
                final_accuracy = float(accuracies[-1])
                print(f"Final training accuracy: {final_accuracy:.4f}")
        
        final_val_accuracy = None
        if 'val_accuracy' in classification_history:
            val_accuracies = classification_history['val_accuracy']
            if val_accuracies:
                final_val_accuracy = float(val_accuracies[-1])
                print(f"Final validation accuracy: {final_val_accuracy:.4f}")
        
        # Prepare results
        results = {
            'job_id': job_id,
            'student_id': student_id,
            'model_urls': model_urls,
            'accuracy': final_accuracy,
            'val_accuracy': final_val_accuracy,
            'training_metrics': {
                'final_accuracy': final_accuracy,
                'final_val_accuracy': final_val_accuracy,
                'classification_history': classification_history,
                'epochs_trained': len(classification_history.get('loss', [])),
                'total_samples': total_samples,
                'students_count': len(processed_data),
                'preprocessing_stats': {
                    'processed_count': preprocessor.processed_count,
                    'error_count': preprocessor.error_count
                }
            }
        }
        
        # Save results to S3
        results_path = '/tmp/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        s3.upload_file(results_path, bucket, f'training_results/{job_id}.json')
        print("Uploaded training results to S3")
        
        print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"Job ID: {job_id}")
        print(f"Final accuracy: {final_accuracy}")
        print(f"Models uploaded: {len(model_urls)} files")
        print(f"Total training samples: {total_samples}")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print("\n=== TRAINING FAILED ===")
        print(f"Error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: train_gpu_tf213.py <training_data_key> <job_id> <student_id>")
        sys.exit(1)
    
    training_data_key = sys.argv[1]
    job_id = sys.argv[2]
    student_id = int(sys.argv[3])
    
    print("Starting GPU training with arguments:")
    print(f"  Training data key: {training_data_key}")
    print(f"  Job ID: {job_id}")
    print(f"  Student ID: {student_id}")
    print(f"  TensorFlow version: {tf.__version__}")
    
    train_on_gpu(training_data_key, job_id, student_id)
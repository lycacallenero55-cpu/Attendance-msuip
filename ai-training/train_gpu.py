#!/usr/bin/env python3
"""
GPU Training Script for Signature AI
Fixed and optimized version for AWS GPU instances
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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/training.log')
    ]
)
logger = logging.getLogger(__name__)

# GPU Configuration
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), using GPU acceleration")
            return True
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
            return False
    else:
        logger.warning("No GPU found, using CPU")
        return False

# Initialize GPU
gpu_available = configure_gpu()

class SignaturePreprocessor:
    """Enhanced signature preprocessor with better error handling"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.processed_count = 0
        self.error_count = 0
        
    def preprocess_signature(self, img_data, debug_name="unknown"):
        """Preprocess signature image with comprehensive error handling"""
        try:
            img = None
            
            # Handle different input formats
            if isinstance(img_data, dict):
                # New wrapped format: {array: [...], shape: [...]} or {base64: "..."}
                if 'base64' in img_data:
                    try:
                        b = base64.b64decode(img_data['base64'])
                        img = Image.open(io.BytesIO(b))
                        logger.debug(f"Loaded base64 image for {debug_name}")
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 for {debug_name}: {e}")
                        return None
                elif 'array' in img_data:
                    arr = np.array(img_data['array'], dtype=np.float32)
                    shape = img_data.get('shape') or []
                    
                    # Try to reshape using provided shape
                    try:
                        if shape:
                            arr = arr.reshape(shape)
                        # If still flat, try to square or 3-channel heuristics
                        if arr.ndim == 1:
                            total = arr.size
                            side = int(np.sqrt(total))
                            if side*side == total:
                                arr = arr.reshape(side, side)
                            elif (total % 3) == 0:
                                side = int(np.sqrt(total//3))
                                if side*side*3 == total:
                                    arr = arr.reshape(side, side, 3)
                        if arr.ndim == 2:
                            arr = np.stack([arr]*3, axis=-1)
                        elif arr.ndim == 3 and arr.shape[2] == 1:
                            arr = np.repeat(arr, 3, axis=2)
                        img = Image.fromarray(np.clip(arr*255 if arr.max()<=1.0 else arr, 0, 255).astype(np.uint8))
                        logger.debug(f"Reconstructed image from wrapped array for {debug_name} with shape {arr.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to reconstruct image for {debug_name}: {e}")
                        return None
                else:
                    logger.warning(f"Unknown wrapped dict for {debug_name}")
                    return None
                    
            elif isinstance(img_data, str):
                try:
                    if img_data.startswith('data:'):
                        img_data = img_data.split(',')[1]
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    logger.debug(f"Successfully loaded base64 image for {debug_name}")
                except Exception as e:
                    logger.warning(f"Failed to decode base64 for {debug_name}: {e}")
                    return None
                    
            elif isinstance(img_data, list):
                img_array = np.array(img_data, dtype=np.float32)
                logger.debug(f"Processing array data for {debug_name}, shape: {img_array.shape}")
                
                # Handle different array shapes
                if len(img_array.shape) == 1:
                    total_pixels = len(img_array)
                    side = int(np.sqrt(total_pixels))
                    if side * side == total_pixels:
                        img_array = img_array.reshape(side, side)
                        logger.debug(f"Reshaped flat array to {img_array.shape}")
                    else:
                        # Try common image sizes
                        common_sizes = [(224,224), (256,256), (128,128), (64,64)]
                        reshaped = False
                        for h, w in common_sizes:
                            if h*w == total_pixels:
                                img_array = img_array.reshape(h, w)
                                logger.debug(f"Reshaped to common size: {img_array.shape}")
                                reshaped = True
                                break
                        if not reshaped:
                            for h, w in common_sizes:
                                if h*w*3 == total_pixels:
                                    img_array = img_array.reshape(h, w, 3)
                                    logger.debug(f"Reshaped to 3-channel: {img_array.shape}")
                                    reshaped = True
                                    break
                        if not reshaped:
                            logger.warning(f"Cannot reshape array of size {total_pixels} to known image format")
                            return None
                            
                elif len(img_array.shape) == 3:
                    if img_array.shape[0] in [1, 3, 4]:
                        img_array = np.transpose(img_array, (1, 2, 0))
                        logger.debug(f"Transposed from CHW to HWC: {img_array.shape}")
                    elif img_array.shape[2] not in [1, 3, 4]:
                        logger.warning(f"Unexpected shape: {img_array.shape}")
                        return None
                
                # Normalize pixel values
                if img_array.max() <= 1.0 and img_array.min() >= 0.0:
                    img_array = (img_array * 255).astype(np.uint8)
                    logger.debug("Scaled normalized values to 0-255 range")
                elif img_array.max() > 255 or img_array.min() < 0:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    logger.debug("Clamped values to 0-255 range")
                else:
                    img_array = img_array.astype(np.uint8)
                
                # Convert to RGB if needed
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                    logger.debug(f"Converted grayscale to RGB: {img_array.shape}")
                elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                    logger.debug(f"Converted single channel to RGB: {img_array.shape}")
                
                try:
                    img = Image.fromarray(img_array)
                    logger.debug(f"Created PIL image from array: {img.size}")
                except Exception as e:
                    logger.warning(f"Failed to create PIL image: {e}")
                    return None
                    
            elif hasattr(img_data, 'size'):
                img = img_data
                logger.debug(f"Using existing PIL image for {debug_name}: {img.size}")
            else:
                logger.warning(f"Unknown image data type for {debug_name}: {type(img_data)}")
                return None
            
            if img is None:
                return None
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                logger.debug("Converted to RGB mode")
            
            # Resize image
            original_size = img.size
            try:
                # Handle different PIL versions
                try:
                    resample_filter = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_filter = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC
                    
                img = img.resize(self.target_size, resample_filter)
                logger.debug(f"Resized from {original_size} to {img.size}")
            except Exception as e:
                logger.warning(f"Failed to resize image: {e}")
                return None
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            self.processed_count += 1
            return img_array
            
        except Exception as e:
            logger.error(f"Error processing {debug_name}: {str(e)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            self.error_count += 1
            return None

class SignatureEmbeddingModel:
    """Enhanced signature embedding model with better error handling"""
    
    def __init__(self, max_students=150):
        self.max_students = max_students
        self.embedding_dim = 128
        self.student_to_id = {}
        self.id_to_student = {}
        self.embedding_model = None
        self.classification_head = None
        self.siamese_model = None
        
    def train_models(self, training_data, epochs=25, validation_split=0.2):
        """Train the signature embedding models"""
        logger.info("Starting model training...")
        
        all_images = []
        all_labels = []
        
        logger.info(f"Processing {len(training_data)} students...")
        
        for idx, (student_name, data) in enumerate(training_data.items()):
            self.student_to_id[student_name] = idx
            self.id_to_student[idx] = student_name
            
            genuine_count = len(data.get('genuine', []))
            forged_count = len(data.get('forged', []))
            
            logger.info(f"Student {student_name} (ID: {idx}): {genuine_count} genuine, {forged_count} forged")
            
            # Process genuine images
            for i, img in enumerate(data.get('genuine', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(idx)
                else:
                    logger.warning(f"Skipping None genuine image {i} for {student_name}")
            
            # Process forged images
            for i, img in enumerate(data.get('forged', [])):
                if img is not None:
                    all_images.append(img)
                    all_labels.append(idx)
                else:
                    logger.warning(f"Skipping None forged image {i} for {student_name}")
        
        logger.info(f"Total images for training: {len(all_images)}")
        logger.info(f"Unique student IDs: {len(set(all_labels))}")
        
        if len(all_images) == 0:
            raise ValueError("No valid training samples found after processing")
        
        if len(all_images) < 5:
            logger.warning("Very few samples for training. Results may be poor.")
            validation_split = 0.0
        
        # Convert to numpy arrays
        logger.info("Converting to numpy arrays...")
        X = np.array(all_images)
        y = keras.utils.to_categorical(all_labels, num_classes=len(training_data))
        
        logger.info(f"Final training data shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Data type: {X.dtype}, range: [{float(X.min()):.3f}, {float(X.max()):.3f}]")
        
        # Build model architecture
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(224, 224, 3)),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(training_data), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model summary:")
        model.summary()
        
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
        
        # Custom callback for progress logging
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logger.info(f"Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, "
                          f"accuracy={logs.get('accuracy', 0):.4f}, "
                          f"val_loss={logs.get('val_loss', 0):.4f}, "
                          f"val_accuracy={logs.get('val_accuracy', 0):.4f}")
        
        callbacks.append(ProgressCallback())
        
        # Train model
        logger.info(f"Starting training with validation_split={validation_split}")
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
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
            raise
        
        # Store models
        self.classification_head = model
        self.embedding_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[-3].output
        )
        
        logger.info("Training completed successfully!")
        return {'classification_history': history.history, 'siamese_history': {}}
    
    def save_models(self, base_path):
        """Save trained models to disk"""
        try:
            if self.embedding_model is not None:
                emb_path = f"{base_path}_embedding.keras"
                self.embedding_model.save(emb_path)
                logger.info(f"Saved embedding model to {emb_path}")
            
            if self.classification_head is not None:
                cls_path = f"{base_path}_classification.keras"
                self.classification_head.save(cls_path)
                logger.info(f"Saved classification model to {cls_path}")
            
            # Save mappings
            map_path = f"{base_path}_mappings.json"
            with open(map_path, 'w') as f:
                json.dump({
                    'student_to_id': self.student_to_id,
                    'id_to_student': {
                        int(k) if isinstance(k, str) and k.isdigit() else k: v
                        for k, v in self.id_to_student.items()
                    }
                }, f, indent=2)
            logger.info(f"Saved mappings to {map_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            traceback.print_exc()
            raise

def train_on_gpu(training_data_key, job_id, student_id):
    """Main training function for GPU instances"""
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        bucket = os.environ.get('S3_BUCKET', 'signatureai-uploads')
        
        logger.info(f"Starting training for job {job_id}")
        logger.info(f"S3 bucket: {bucket}")
        logger.info(f"Training data key: {training_data_key}")
        
        # Download training data
        logger.info(f"Downloading training data from s3://{bucket}/{training_data_key}")
        try:
            response = s3.get_object(Bucket=bucket, Key=training_data_key)
            training_data_raw = json.loads(response['Body'].read())
            logger.info("Successfully downloaded training data")
        except Exception as e:
            logger.error(f"Failed to download training data: {e}")
            traceback.print_exc()
            raise
        
        logger.info(f"Raw training data contains {len(training_data_raw)} students")
        
        # Initialize preprocessor and model
        preprocessor = SignaturePreprocessor(target_size=(224, 224))
        model_manager = SignatureEmbeddingModel(max_students=150)
        
        # Process training data
        logger.info("Processing training data...")
        processed_data = {}
        
        for student_name, data in training_data_raw.items():
            logger.info(f"\nProcessing student: {student_name}")
            
            genuine_images = []
            forged_images = []
            
            # Process genuine images
            genuine_raw = data.get('genuine', [])
            logger.info(f"  Found {len(genuine_raw)} genuine images")
            
            for i, img_data in enumerate(genuine_raw):
                logger.info(f"    Processing genuine image {i+1}/{len(genuine_raw)}")
                processed_img = preprocessor.preprocess_signature(
                    img_data, 
                    debug_name=f"{student_name}_genuine_{i}"
                )
                if processed_img is not None:
                    genuine_images.append(processed_img)
                    logger.info("      ✓ Successfully processed")
                else:
                    logger.warning("      ✗ Failed to process")
            
            # Process forged images
            forged_raw = data.get('forged', [])
            logger.info(f"  Found {len(forged_raw)} forged images")
            
            for i, img_data in enumerate(forged_raw):
                logger.info(f"    Processing forged image {i+1}/{len(forged_raw)}")
                processed_img = preprocessor.preprocess_signature(
                    img_data,
                    debug_name=f"{student_name}_forged_{i}"
                )
                if processed_img is not None:
                    forged_images.append(processed_img)
                    logger.info("      ✓ Successfully processed")
                else:
                    logger.warning("      ✗ Failed to process")
            
            processed_data[student_name] = {
                'genuine': genuine_images,
                'forged': forged_images
            }
            
            total_processed = len(genuine_images) + len(forged_images)
            total_raw = len(genuine_raw) + len(forged_raw)
            success_rate = (total_processed / total_raw * 100) if total_raw > 0 else 0
            
            logger.info(f"  Final: {len(genuine_images)} genuine, {len(forged_images)} forged")
            logger.info(f"  Success rate: {total_processed}/{total_raw} ({success_rate:.1f}%)")
        
        # Print preprocessing summary
        logger.info("\n=== PREPROCESSING SUMMARY ===")
        logger.info(f"Total images processed successfully: {preprocessor.processed_count}")
        logger.info(f"Total processing errors: {preprocessor.error_count}")
        
        total_samples = sum(len(d['genuine']) + len(d['forged']) for d in processed_data.values())
        logger.info(f"Total processed samples available for training: {total_samples}")
        
        if total_samples == 0:
            raise ValueError("No valid training samples found after processing")
        
        validation_split = 0.0 if total_samples < 5 else 0.2
        
        # Start model training
        logger.info("\n=== STARTING MODEL TRAINING ===")
        training_result = model_manager.train_models(
            processed_data, 
            epochs=25, 
            validation_split=validation_split
        )
        
        # Save models
        logger.info("Training completed! Saving models...")
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
                logger.info(f"Uploaded {model_type} model to S3: {s3_key}")
            else:
                logger.warning(f"WARNING: {model_type} model file not found: {file_path}")
        
        # Upload mappings
        mappings_path = f'{temp_dir}/signature_model_mappings.json'
        if os.path.exists(mappings_path):
            s3_key = f'models/{job_id}/mappings.json'
            s3.upload_file(mappings_path, bucket, s3_key)
            model_urls['mappings'] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
            logger.info(f"Uploaded mappings to S3: {s3_key}")
        
        # Save SavedModel format for better compatibility
        try:
            savedmodel_dir = f'{temp_dir}/{job_id}_classification.tf'
            model_manager.classification_head.save(savedmodel_dir, save_format='tf')
            logger.info(f"Saved classifier SavedModel to {savedmodel_dir}")
            
            # Create zip file
            import zipfile
            zip_path = f'{temp_dir}/{job_id}_classification_savedmodel.zip'
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(savedmodel_dir):
                    for f in files:
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, start=savedmodel_dir)
                        zf.write(full_path, arcname=rel_path)
            
            s3_key = f'models/{job_id}/classifier_savedmodel.zip'
            s3.upload_file(zip_path, bucket, s3_key)
            model_urls['classifier_savedmodel_zip'] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
            logger.info(f"Uploaded classifier SavedModel zip to S3: {s3_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save/upload classifier SavedModel: {e}")
        
        # Create classifier spec
        try:
            spec = {
                'type': 'classification',
                'backbone': 'SimpleCNN',
                'image_size': 224,
                'num_classes': int(len(model_manager.id_to_student)),
            }
            spec_path = f'{temp_dir}/classifier_spec.json'
            with open(spec_path, 'w') as f:
                json.dump(spec, f, indent=2)
            
            s3_key = f'models/{job_id}/classifier_spec.json'
            s3.upload_file(spec_path, bucket, s3_key)
            model_urls['classifier_spec'] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
            logger.info(f"Uploaded classifier_spec.json to S3: {s3_key}")
            
        except Exception as e:
            logger.warning(f"Failed to write/upload classifier spec: {e}")
        
        # Compute and upload centroids
        try:
            centroids = {}
            for cls_idx, student_name in model_manager.id_to_student.items():
                try:
                    imgs = processed_data.get(student_name, {}).get('genuine', [])
                    if not imgs:
                        continue
                    arr = np.array(imgs, dtype=np.float32)
                    embs = model_manager.embedding_model.predict(arr, verbose=0)
                    centroids[str(int(cls_idx))] = np.mean(embs, axis=0).tolist()
                except Exception as e:
                    logger.warning(f"Failed to compute centroid for class {cls_idx}: {e}")
                    continue
            
            if centroids:
                cpath = f'{temp_dir}/centroids.json'
                with open(cpath, 'w') as f:
                    json.dump(centroids, f, indent=2)
                
                s3_key = f'models/{job_id}/centroids.json'
                s3.upload_file(cpath, bucket, s3_key)
                model_urls['centroids'] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
                logger.info(f"Uploaded centroids.json to S3: {s3_key}")
                
        except Exception as e:
            logger.warning(f"Failed to compute/upload centroids: {e}")
        
        # Extract training metrics
        classification_history = training_result.get('classification_history', {})
        final_accuracy = None
        if 'accuracy' in classification_history:
            accuracies = classification_history['accuracy']
            if accuracies:
                final_accuracy = float(accuracies[-1])
                logger.info(f"Final training accuracy: {final_accuracy:.4f}")
        
        final_val_accuracy = None
        if 'val_accuracy' in classification_history:
            val_accuracies = classification_history['val_accuracy']
            if val_accuracies:
                final_val_accuracy = float(val_accuracies[-1])
                logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")
        
        # Create results
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
        
        # Ensure JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(v) for v in obj]
            return obj
        
        results = make_json_serializable(results)
        
        # Upload results
        results_path = '/tmp/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        s3.upload_file(results_path, bucket, f'training_results/{job_id}.json')
        logger.info("Uploaded training results to S3")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Final summary
        logger.info("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Final accuracy: {final_accuracy}")
        logger.info(f"Models uploaded: {len(model_urls)} files")
        logger.info(f"Total training samples: {total_samples}")
        
        return results
        
    except Exception as e:
        logger.error("\n=== TRAINING FAILED ===")
        logger.error(f"Error: {str(e)}")
        logger.error("Full traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Usage: train_gpu.py <training_data_key> <job_id> <student_id>")
        sys.exit(1)
    
    training_data_key = sys.argv[1]
    job_id = sys.argv[2]
    student_id = int(sys.argv[3])
    
    logger.info("Starting GPU training with arguments:")
    logger.info(f"  Training data key: {training_data_key}")
    logger.info(f"  Job ID: {job_id}")
    logger.info(f"  Student ID: {student_id}")
    logger.info(f"  TensorFlow version: {tf.__version__}")
    logger.info(f"  GPU available: {gpu_available}")
    
    try:
        result = train_on_gpu(training_data_key, job_id, student_id)
        logger.info("Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
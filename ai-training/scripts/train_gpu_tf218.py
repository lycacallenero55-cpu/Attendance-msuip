#!/usr/bin/env python3
"""
GPU Training Script - TensorFlow 2.18 Compatible
Global Signature Classifier for Multi-Student Verification
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
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.global_signature_classifier import GlobalSignatureClassifier

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

def process_training_data_for_global_model(training_data_raw, preprocessor):
    """
    Process training data for Global Signature Classifier
    Returns data in format: {student_id: [images]}
    """
    print("Processing training data for Global Signature Classifier...")
    processed_data = {}
    
    for student_name, data in training_data_raw.items():
        print(f"\nProcessing student: {student_name}")
        student_images = []
        
        # Process genuine images only (no forgery detection)
        genuine_raw = data.get('genuine', [])
        print(f"  Found {len(genuine_raw)} genuine images")
        
        for i, img_data in enumerate(genuine_raw):
            print(f"    Processing genuine image {i+1}/{len(genuine_raw)}")
            processed_img = preprocessor.preprocess_signature(
                img_data, 
                debug_name=f"{student_name}_genuine_{i}"
            )
            if processed_img is not None:
                student_images.append(processed_img)
                print("      ✓ Successfully processed")
            else:
                print("      ✗ Failed to process")
        
        if student_images:
            processed_data[student_name] = student_images
            print(f"  Final: {len(student_images)} valid images for {student_name}")
        else:
            print(f"  ⚠️  No valid images for {student_name}, skipping")
    
    print(f"\nProcessed {len(processed_data)} students with valid data")
    return processed_data

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
        
        # Process training data for Global Signature Classifier
        processed_data = process_training_data_for_global_model(training_data_raw, preprocessor)
        
        print("\n=== PREPROCESSING SUMMARY ===")
        print(f"Total images processed successfully: {preprocessor.processed_count}")
        print(f"Total processing errors: {preprocessor.error_count}")
        
        total_samples = sum(len(images) for images in processed_data.values())
        print(f"Total processed samples available for training: {total_samples}")
        
        if total_samples == 0:
            raise ValueError("No valid training samples found after processing")
        
        # Initialize Global Signature Classifier
        print("\n=== INITIALIZING GLOBAL SIGNATURE CLASSIFIER ===")
        global_model = GlobalSignatureClassifier(
            image_size=224,
            embedding_dim=512,
            learning_rate=0.001,
            max_students=1000
        )
        
        # Add all students to the global model
        for student_name in processed_data.keys():
            global_model.add_student(student_name, student_name)
        
        print(f"Added {global_model.num_classes} students to global model")
        
        # Train the global model
        print("\n=== STARTING GLOBAL MODEL TRAINING ===")
        training_history = global_model.train_global_model(
            training_data=processed_data,
            epochs=50,
            validation_split=0.2
        )
        
        print("Training completed! Saving global model...")
        temp_dir = f'/tmp/{job_id}_models'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save global model
        model_path = f'{temp_dir}/global_signature_model'
        global_model.save_model(model_path)
        
        # Upload model to S3
        model_files = [
            f"{model_path}.keras",
            f"{model_path}_metadata.json"
        ]
        
        model_urls = {}
        for file_path in model_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                s3_key = f"models/global/{job_id}/{filename}"
                s3.upload_file(file_path, bucket, s3_key)
                model_urls[filename] = f'https://{bucket}.s3.amazonaws.com/{s3_key}'
                print(f"Uploaded {filename} to S3: {s3_key}")
            else:
                print(f"WARNING: Model file not found: {file_path}")
        
        # Extract accuracy metrics from training history
        final_accuracy = None
        if 'accuracy' in training_history:
            accuracies = training_history['accuracy']
            if accuracies:
                final_accuracy = float(accuracies[-1])
                print(f"Final training accuracy: {final_accuracy:.4f}")
        
        final_val_accuracy = None
        if 'val_accuracy' in training_history:
            val_accuracies = training_history['val_accuracy']
            if val_accuracies:
                final_val_accuracy = float(val_accuracies[-1])
                print(f"Final validation accuracy: {final_val_accuracy:.4f}")
        
        # Prepare results
        results = {
            'job_id': job_id,
            'student_id': student_id,
            'model_type': 'global_classifier',
            'model_urls': model_urls,
            'accuracy': final_accuracy,
            'val_accuracy': final_val_accuracy,
            'training_metrics': {
                'final_accuracy': final_accuracy,
                'final_val_accuracy': final_val_accuracy,
                'training_history': training_history,
                'epochs_trained': len(training_history.get('loss', [])),
                'total_samples': total_samples,
                'students_count': len(processed_data),
                'num_classes': global_model.num_classes,
                'students': list(global_model.student_names.keys()),
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
        s3.upload_file(results_path, bucket, f'training_results/global/{job_id}.json')
        print("Uploaded training results to S3")
        
        print("\n=== GLOBAL SIGNATURE CLASSIFIER TRAINING COMPLETED ===")
        print(f"Job ID: {job_id}")
        print(f"Model Type: Global Classifier")
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")
        print(f"Number of students: {global_model.num_classes}")
        print(f"Students: {list(global_model.student_names.keys())}")
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
        print("Usage: train_gpu_tf218.py <training_data_key> <job_id> <student_id>")
        sys.exit(1)
    
    training_data_key = sys.argv[1]
    job_id = sys.argv[2]
    student_id = int(sys.argv[3])
    
    print("🎓 Global Signature Classifier GPU Training")
    print("=" * 50)
    print(f"Training data key: {training_data_key}")
    print(f"Job ID: {job_id}")
    print(f"Student ID: {student_id}")
    print(f"TensorFlow version: {tf.__version__}")
    print("=" * 50)
    
    train_on_gpu(training_data_key, job_id, student_id)
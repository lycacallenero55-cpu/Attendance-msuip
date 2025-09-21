#!/usr/bin/env python3
"""
Global Signature Classifier GPU Training Script
Multi-student signature classification with GPU acceleration
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
import zipfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.global_signature_classifier import GlobalSignatureClassifier
from utils.signature_preprocessing import SignaturePreprocessor

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

def download_training_data(s3_client, bucket_name: str, training_data_key: str) -> dict:
    """Download training data from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=training_data_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Downloaded training data: {len(data)} students")
        return data
    except Exception as e:
        print(f"Error downloading training data: {e}")
        raise

def preprocess_images(images_data: list, preprocessor: SignaturePreprocessor) -> list:
    """Preprocess signature images"""
    processed_images = []
    
    for image_data in images_data:
        try:
            # Decode base64 image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = image_data
            
            # Preprocess image
            processed = preprocessor.preprocess_signature(image)
            processed_images.append(processed)
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            continue
    
    return processed_images

def train_global_model(training_data: dict, job_id: str, student_id: str = None) -> dict:
    """
    Train the Global Signature Classifier
    """
    print("🚀 Starting Global Signature Classifier Training")
    print(f"Job ID: {job_id}")
    print(f"Student ID: {student_id}")
    
    try:
        # Initialize preprocessor
        preprocessor = SignaturePreprocessor(target_size=224)
        
        # Initialize Global model
        model = GlobalSignatureClassifier(
            image_size=224,
            embedding_dim=512,
            learning_rate=0.001,
            max_students=1000
        )
        
        # Prepare training data
        print("📊 Preparing training data...")
        processed_training_data = {}
        
        for student_id_key, images_data in training_data.items():
            print(f"Processing student {student_id_key}: {len(images_data)} images")
            
            # Preprocess images
            processed_images = preprocess_images(images_data, preprocessor)
            
            if processed_images:
                processed_training_data[student_id_key] = processed_images
                print(f"✅ Processed {len(processed_images)} images for student {student_id_key}")
            else:
                print(f"⚠️  No valid images for student {student_id_key}")
        
        if not processed_training_data:
            raise ValueError("No valid training data found")
        
        # Train global model
        print("🎯 Training global classifier...")
        history = model.train_global_model(
            training_data=processed_training_data,
            epochs=50,
            validation_split=0.2
        )
        
        # Get training results
        final_accuracy = history.get('accuracy', [0])[-1]
        final_val_accuracy = history.get('val_accuracy', [0])[-1]
        
        print(f"✅ Training completed!")
        print(f"Final accuracy: {final_accuracy:.4f}")
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")
        print(f"Number of students: {model.num_classes}")
        
        # Save model locally
        model_path = f"/tmp/global_model_{job_id}"
        model.save_model(model_path)
        
        # Upload model to S3
        s3_client = boto3.client('s3')
        bucket_name = os.getenv('S3_BUCKET', 'your-bucket-name')
        
        # Upload model files
        model_files = [
            f"{model_path}.keras",
            f"{model_path}_metadata.json"
        ]
        
        uploaded_files = {}
        for file_path in model_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                s3_key = f"models/global/{job_id}/{filename}"
                
                s3_client.upload_file(file_path, bucket_name, s3_key)
                uploaded_files[filename] = s3_key
                print(f"✅ Uploaded {filename} to S3")
        
        # Create training results
        results = {
            "job_id": job_id,
            "model_type": "global_classifier",
            "training_status": "completed",
            "final_accuracy": float(final_accuracy),
            "final_val_accuracy": float(final_val_accuracy),
            "num_students": model.num_classes,
            "students": list(model.student_names.keys()),
            "model_files": uploaded_files,
            "training_history": {
                "accuracy": [float(x) for x in history.get('accuracy', [])],
                "val_accuracy": [float(x) for x in history.get('val_accuracy', [])],
                "loss": [float(x) for x in history.get('loss', [])],
                "val_loss": [float(x) for x in history.get('val_loss', [])]
            }
        }
        
        # Save results to S3
        results_key = f"training_results/global/{job_id}/results.json"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=results_key,
            Body=json.dumps(results, indent=2),
            ContentType='application/json'
        )
        
        print(f"✅ Results saved to S3: {results_key}")
        
        # Cleanup
        for file_path in model_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return results
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Save error results
        error_results = {
            "job_id": job_id,
            "model_type": "global_classifier",
            "training_status": "failed",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }
        
        try:
            s3_client = boto3.client('s3')
            bucket_name = os.getenv('S3_BUCKET', 'your-bucket-name')
            error_key = f"training_results/global/{job_id}/error.json"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=error_key,
                Body=json.dumps(error_results, indent=2),
                ContentType='application/json'
            )
        except:
            pass
        
        raise

def main():
    """Main training function"""
    if len(sys.argv) != 4:
        print("Usage: train_global_gpu.py <training_data_key> <job_id> <student_id>")
        sys.exit(1)
    
    training_data_key = sys.argv[1]
    job_id = sys.argv[2]
    student_id = sys.argv[3]
    
    print("🎓 Global Signature Classifier GPU Training")
    print("=" * 50)
    print(f"Training data key: {training_data_key}")
    print(f"Job ID: {job_id}")
    print(f"Student ID: {student_id}")
    print("=" * 50)
    
    try:
        # Download training data
        s3_client = boto3.client('s3')
        bucket_name = os.getenv('S3_BUCKET', 'your-bucket-name')
        
        print("📥 Downloading training data...")
        training_data = download_training_data(s3_client, bucket_name, training_data_key)
        
        # Train model
        results = train_global_model(training_data, job_id, student_id)
        
        print("🎉 Training completed successfully!")
        print(f"Results: {json.dumps(results, indent=2)}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
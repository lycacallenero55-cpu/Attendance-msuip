from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import os
import uuid
from datetime import datetime
import time
import logging
import asyncio

from models.database import db_manager
from models.signature_model import SignatureVerificationModel
from utils.image_processing import preprocess_image
from utils.storage import save_to_supabase, cleanup_local_file
from utils.augmentation import SignatureAugmentation
from utils.job_queue import job_queue
from utils.training_callback import RealTimeMetricsCallback
from services.model_versioning import model_versioning_service
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance
model_manager = SignatureVerificationModel()

async def train_signature_model(student, genuine_data, forged_data, job=None):
    """
    Unified training function for both sync and async routes
    """
    try:
        # Process and validate images
        genuine_images = []
        forged_images = []
        
        # Process genuine images
        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            processed_image = preprocess_image(image, settings.MODEL_IMAGE_SIZE)
            genuine_images.append(processed_image)
            
            # Update progress if job provided
            if job:
                progress = 5.0 + (i + 1) / len(genuine_data) * 15.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing genuine images... {i+1}/{len(genuine_data)}")
        
        # Process forged images
        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            processed_image = preprocess_image(image, settings.MODEL_IMAGE_SIZE)
            forged_images.append(processed_image)
            
            # Update progress if job provided
            if job:
                progress = 20.0 + (i + 1) / len(forged_data) * 15.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing forged images... {i+1}/{len(forged_data)}")
        
        # Apply moderate data augmentation
        if job:
            job_queue.update_job_progress(job.job_id, 35.0, "Applying moderate data augmentation...")
        
        augmenter = SignatureAugmentation(
            rotation_range=8.0,  # Reduced from 15.0
            scale_range=(0.95, 1.05),  # Reduced from (0.9, 1.1)
            brightness_range=0.2,  # Reduced from 0.3
            blur_probability=0.2,  # Reduced from 0.3
            thickness_variation=0.05,  # Reduced from 0.1
            elastic_alpha=4.0,  # Reduced from 8.0
            elastic_sigma=2.0,  # Reduced from 4.0
            noise_stddev=3.0,  # Reduced from 5.0
            shear_range=0.1,  # Reduced from 0.2
            perspective_distortion=0.02,  # Reduced from 0.03
            camera_tilt_range=5.0,  # Reduced from 10.0
            lighting_angle_range=10.0  # Reduced from 20.0
        )
        
        genuine_augmented, genuine_labels = augmenter.augment_batch(
            genuine_images, [True] * len(genuine_images), augmentation_factor=2  # Reduced from 3
        )
        
        forged_augmented, forged_labels = augmenter.augment_batch(
            forged_images, [False] * len(forged_images), augmentation_factor=1  # Reduced from 2
        )
        
        all_images = genuine_augmented + forged_augmented
        all_labels = genuine_labels + forged_labels
        
        # Create model record in database
        if job:
            job_queue.update_job_progress(job.job_id, 40.0, "Creating model record...")
        
        model_uuid = str(uuid.uuid4())
        model_data = {
            "student_id": int(student["id"]),
            "model_path": f"models/{model_uuid}.keras",
            "status": "training",
            "sample_count": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images),
            "training_date": datetime.utcnow().isoformat()
        }
        created = await db_manager.create_trained_model(model_data)
        numeric_model_id = created["id"] if isinstance(created, dict) else None
        
        # Start training
        if job:
            job_queue.update_job_progress(job.job_id, 45.0, "Initializing AI model...")
        
        model_manager = SignatureVerificationModel()
        
        if job:
            job_queue.update_job_progress(job.job_id, 50.0, "Training AI model...")
        
        # Create real-time metrics callback
        realtime_callback = RealTimeMetricsCallback(job.job_id, settings.MODEL_EPOCHS) if job else None
        if realtime_callback:
            logger.info(f"Created RealTimeMetricsCallback for job {job.job_id} with {settings.MODEL_EPOCHS} epochs")
        
        t0 = time.time()
        logger.info(f"Starting training with {len(all_images)} images and custom callbacks: {realtime_callback is not None}")
        history = model_manager.train_with_augmented_data(
            all_images, 
            all_labels, 
            custom_callbacks=[realtime_callback] if realtime_callback else None
        )
        
        # Compute prototype and threshold
        if job:
            job_queue.update_job_progress(job.job_id, 80.0, "Computing prototype and threshold...")
        
        # Split into train/validation for threshold computation
        # Use larger validation set for more reliable evaluation
        from sklearn.model_selection import train_test_split
        train_genuine, val_genuine = train_test_split(
            genuine_images, test_size=0.3, random_state=42, stratify=None
        )
        if forged_images:
            train_forged, val_forged = train_test_split(
                forged_images, test_size=0.3, random_state=42, stratify=None
            )
        else:
            train_forged, val_forged = [], []
        
        centroid, threshold = model_manager.compute_centroid_and_adaptive_threshold(
            train_genuine,
            train_forged if len(train_forged) > 0 else None,
            val_genuine,
            val_forged if len(val_forged) > 0 else None
        )
        
        # Save models
        if job:
            job_queue.update_job_progress(job.job_id, 85.0, "Saving model...")
        
        local_model_path = os.path.join(settings.LOCAL_MODELS_DIR, f"{model_uuid}.keras")
        local_embed_path = os.path.join(settings.LOCAL_MODELS_DIR, f"{model_uuid}_embed.keras")
        
        model_manager.save_model(local_model_path)
        model_manager.save_embedding_model(local_embed_path)
        
        # Upload to Supabase
        if job:
            job_queue.update_job_progress(job.job_id, 90.0, "Uploading to cloud storage...")
        
        remote_model_path = await save_to_supabase(local_model_path, f"models/{model_uuid}.keras")
        remote_embed_path = await save_to_supabase(local_embed_path, f"models/{model_uuid}_embed.keras")
        
        # Update model metadata
        if numeric_model_id:
            await db_manager.update_model_metadata(numeric_model_id, {
                "status": "completed",
                "accuracy": float(history.history.get("val_accuracy", [0])[-1]),
                "prototype_centroid": centroid if isinstance(centroid, list) else centroid.tolist(),
                "prototype_threshold": float(threshold),
                "embedding_model_path": remote_embed_path,
                "model_path": remote_model_path
            })
        
        # Cleanup local files
        cleanup_local_file(local_model_path)
        cleanup_local_file(local_embed_path)
        
        # Calculate training metrics
        train_time = time.time() - t0
        val_accuracy = history.history.get("val_accuracy", [0])[-1]
        precision = history.history.get("val_precision", [0])[-1] if "val_precision" in history.history else 0
        recall = history.history.get("val_recall", [0])[-1] if "val_recall" in history.history else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        result = {
            "success": True,
            "model_id": numeric_model_id,
            "model_uuid": model_uuid,
            "accuracy": float(history.history.get("accuracy", [0])[-1]),
            "val_accuracy": float(val_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "train_time_s": float(train_time),
            "threshold": float(threshold),
            "training_samples": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images)
        }
        
        if job:
            job_queue.complete_job(job.job_id, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if job:
            job_queue.fail_job(job.job_id, str(e))
        raise

@router.post("/start")
async def start_training(
    student_id: str = Form(...),
    genuine_files: List[UploadFile] = File(...),
    forged_files: List[UploadFile] = File(...)
):
    """Start training a signature verification model for a student"""
    
    try:
        # Validate student exists by school/student_id string; fallback to numeric id
        student = await db_manager.get_student_by_school_id(student_id)
        if not student:
            try:
                numeric_id = int(student_id)
                student = await db_manager.get_student(numeric_id)
            except Exception:
                student = None
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Validate minimum samples
        if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
            raise HTTPException(
                status_code=400, 
                detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required"
            )
        
        if len(forged_files) < settings.MIN_FORGED_SAMPLES:
            raise HTTPException(
                status_code=400, 
                detail=f"Minimum {settings.MIN_FORGED_SAMPLES} forged samples required"
            )
        
        # Read file data
        genuine_data = []
        for file in genuine_files:
            data = await file.read()
            genuine_data.append(data)
        
        forged_data = []
        for file in forged_files:
            data = await file.read()
            forged_data.append(data)
        
        # Use unified training function
        result = await train_signature_model(student, genuine_data, forged_data)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status/{model_id}")
async def get_training_status(model_id: str):
    """Get the training status of a model"""
    try:
        models = await db_manager.get_trained_models()
        model = next((m for m in models if m["id"] == model_id), None)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "model_id": model_id,
            "status": model["status"],
            "accuracy": model.get("accuracy"),
            "training_date": model["training_date"],
            "sample_count": model["sample_count"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# In training.py, update the start_async_training function:

@router.post("/start-async")
async def start_async_training(
    student_id: str = Form(...),
    genuine_files: List[UploadFile] = File(...),
    forged_files: List[UploadFile] = File(...)
):
    """Start async training job for a signature verification model"""
    
    try:
        # Validate student exists
        student = await db_manager.get_student_by_school_id(student_id)
        if not student:
            try:
                numeric_id = int(student_id)
                student = await db_manager.get_student(numeric_id)
            except Exception:
                student = None
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Validate minimum samples
        if len(genuine_files) < settings.MIN_GENUINE_SAMPLES:
            raise HTTPException(
                status_code=400, 
                detail=f"Minimum {settings.MIN_GENUINE_SAMPLES} genuine samples required"
            )
        
        if len(forged_files) < settings.MIN_FORGED_SAMPLES:
            raise HTTPException(
                status_code=400, 
                detail=f"Minimum {settings.MIN_FORGED_SAMPLES} forged samples required"
            )
        
        # Create async training job
        job = job_queue.create_job(int(student["id"]), "training")
        
        # CRITICAL FIX: Read file contents BEFORE spawning background task
        genuine_data = []
        for f in genuine_files:
            data = await f.read()
            genuine_data.append(data)
        
        forged_data = []
        for f in forged_files:
            data = await f.read()
            forged_data.append(data)
        
        # Now pass the raw data to the background task
        asyncio.create_task(run_async_training(job, student, genuine_data, forged_data))
        
        return {
            "success": True,
            "job_id": job.job_id,
            "message": "Training job started",
            "stream_url": f"/api/progress/stream/{job.job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async training: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
async def run_async_training(job, student, genuine_data, forged_data):
    """Run training job in background with progress updates."""
    try:
        job_queue.start_job(job.job_id)
        
        # Use unified training function with job for progress updates
        result = await train_signature_model(student, genuine_data, forged_data, job)
        
    except Exception as e:
        logger.error(f"Async training failed: {e}")
        job_queue.fail_job(job.job_id, str(e))

@router.get("/models")
async def get_trained_models(student_id: Optional[int] = None):
    """Get all trained models, optionally filtered by student"""
    try:
        models = await db_manager.get_trained_models(student_id)
        return {"models": models}
    
    except Exception as e:
        logger.error(f"Error getting trained models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
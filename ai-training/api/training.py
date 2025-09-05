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
from services.model_versioning import model_versioning_service
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global model instance
model_manager = SignatureVerificationModel()

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
        
        # Process and validate images
        genuine_images = []
        forged_images = []
        
        # Process genuine images
        for file in genuine_files:
            
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image, settings.MODEL_IMAGE_SIZE)
            genuine_images.append(processed_image)
        
        # Process forged images
        for file in forged_files:
            
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image, settings.MODEL_IMAGE_SIZE)
            forged_images.append(processed_image)
        
        # Apply data augmentation to increase training robustness
        logger.info(f"Applying data augmentation to {len(genuine_images)} genuine and {len(forged_images)} forged samples")
        augmenter = SignatureAugmentation(
            rotation_range=15.0,
            scale_range=(0.9, 1.1),
            brightness_range=0.2,
            blur_probability=0.3,
            thickness_variation=0.1
        )
        
        # Augment genuine signatures more aggressively (3x augmentation)
        genuine_augmented, genuine_labels = augmenter.augment_batch(
            genuine_images, [True] * len(genuine_images), augmentation_factor=3
        )
        
        # Augment forged signatures less aggressively (1x augmentation)
        forged_augmented, forged_labels = augmenter.augment_batch(
            forged_images, [False] * len(forged_images), augmentation_factor=1
        )
        
        # Combine all augmented data
        all_images = genuine_augmented + forged_augmented
        all_labels = genuine_labels + forged_labels
        
        logger.info(f"After augmentation: {len(all_images)} total samples ({len(genuine_augmented)} genuine, {len(forged_augmented)} forged)")
        
        # Create model record in database
        model_uuid = str(uuid.uuid4())
        model_data = {
            # Let DB auto-generate bigint primary key id
            "student_id": int(student["id"]) if isinstance(student.get("id"), (int, float)) else student.get("id"),
            "model_path": f"models/{model_uuid}.keras",
            "status": "training",
            "sample_count": len(genuine_images) + len(forged_images),
            "genuine_count": len(genuine_images),
            "forged_count": len(forged_images),
            "training_date": datetime.utcnow().isoformat()
        }
        created = await db_manager.create_trained_model(model_data)
        numeric_model_id = created["id"] if isinstance(created, dict) else None
        
        # Start training (this would be async in production)
        try:
            t0 = time.time()
            # Train with augmented data for better robustness
            history = model_manager.train_with_augmented_data(all_images, all_labels)

            # Compute prototype (centroid) from genuine samples
            centroid, threshold = model_manager.compute_centroid_and_adaptive_threshold(
            genuine_images, 
            forged_images if len(forged_images) > 0 else None
            )
            # Calibrate threshold via EER using distances to centroid
            gen_emb = model_manager.embed_images(genuine_images)
            forg_emb = model_manager.embed_images(forged_images) if len(forged_images) else np.zeros((0, gen_emb.shape[1]))
            centroid_vec = np.array(centroid)
            gen_d = np.linalg.norm(gen_emb - centroid_vec, axis=1)
            forg_d = np.linalg.norm(forg_emb - centroid_vec, axis=1) if forg_emb.size else np.array([])

            def compute_eer_threshold(genuine_d, forged_d):
                if len(genuine_d) == 0:
                    return 0.7, 0.0, 0.0
                # Candidate thresholds are unique distances observed
                all_d = np.concatenate([genuine_d, forged_d]) if len(forged_d) else genuine_d
                all_d = np.unique(np.sort(all_d))
                best_thr = all_d[0] if all_d.size > 0 else 0.7
                best_gap = 1e9
                best_far = 0.0
                best_frr = 0.0
                total_gen = max(1, len(genuine_d))
                total_forg = max(1, len(forged_d))
                for thr in all_d:
                    # Accept if distance <= thr
                    frr = float(np.sum(genuine_d > thr)) / total_gen
                    far = float(np.sum(forged_d <= thr)) / total_forg if total_forg > 0 else 0.0
                    gap = abs(far - frr)
                    if gap < best_gap:
                        best_gap = gap
                        best_thr = thr
                        best_far = far
                        best_frr = frr
                return float(best_thr), float(best_far), float(best_frr)

            threshold, far_at_thr, frr_at_thr = compute_eer_threshold(gen_d, forg_d)
            
            # Get best accuracy
            best_accuracy = max(history.history['val_accuracy'])
            train_time_s = int(time.time() - t0)
            best_precision = max(history.history.get('precision', [0])) if history.history.get('precision') else 0
            best_recall = max(history.history.get('recall', [0])) if history.history.get('recall') else 0
            f1 = (2 * best_precision * best_recall / (best_precision + best_recall)) if (best_precision + best_recall) > 0 else 0
            
            # Save full model (Keras format) and embedding-only model
            local_model_path = os.path.join(settings.LOCAL_MODELS_DIR, f"{model_uuid}.keras")
            model_manager.save_model(local_model_path)

            # Save embedding-only model if available
            embedding_local_path = None
            if getattr(model_manager, 'embedding_model', None) is not None:
                embedding_local_path = os.path.join(settings.LOCAL_MODELS_DIR, f"{model_uuid}_embed.keras")
                model_manager.embedding_model.save(embedding_local_path)
            
            # Upload to Supabase Storage
            supabase_path = await save_to_supabase(local_model_path, f"models/{model_uuid}.keras")
            embedding_supabase_path = None
            if embedding_local_path:
                embedding_supabase_path = await save_to_supabase(embedding_local_path, f"models/{model_uuid}_embed.keras")
            
            # Update model record
            await db_manager.update_model_status(
                numeric_model_id,
                "completed",
                float(best_accuracy)
            )
            # Store prototype metadata and final artifact paths
            await db_manager.update_model_metadata(numeric_model_id, {
                "model_path": supabase_path,
                "prototype_centroid": centroid,
                "prototype_threshold": threshold,
                "embedding_model_path": embedding_supabase_path
            })
            
            # Cleanup local file
            cleanup_local_file(local_model_path)
            if embedding_local_path:
                cleanup_local_file(embedding_local_path)
            
            return {
                "success": True,
                "model_id": numeric_model_id,
                "model_uuid": model_uuid,
                "status": "completed",
                "accuracy": float(best_accuracy),
                "val_accuracy": float(best_accuracy),
                "precision": float(best_precision),
                "recall": float(best_recall),
                "f1": float(f1),
                "training_samples": len(genuine_images) + len(forged_images),
                "genuine_count": len(genuine_images),
                "forged_count": len(forged_images),
                "train_time_s": train_time_s,
                "calibration": {
                    "threshold": float(threshold),
                    "far": float(far_at_thr),
                    "frr": float(frr_at_thr)
                },
                "message": "Model trained successfully"
            }
            
        except Exception as e:
            # Update model status to failed
            try:
                if numeric_model_id is not None:
                    await db_manager.update_model_status(numeric_model_id, "failed")
            except Exception:
                pass
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
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
        job_queue.update_job_progress(job.job_id, 5.0, "Processing images...")
        
        # Process and validate images
        genuine_images = []
        forged_images = []
        
        # Process genuine images (now using raw data, not file objects)
        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            processed_image = preprocess_image(image, settings.MODEL_IMAGE_SIZE)
            genuine_images.append(processed_image)
            
            # Update progress
            progress = 5.0 + (i + 1) / len(genuine_data) * 15.0
            job_queue.update_job_progress(job.job_id, progress, f"Processing genuine images... {i+1}/{len(genuine_data)}")
        
        # Process forged images
        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            processed_image = preprocess_image(image, settings.MODEL_IMAGE_SIZE)
            forged_images.append(processed_image)
            
            # Update progress
            progress = 20.0 + (i + 1) / len(forged_data) * 15.0
            job_queue.update_job_progress(job.job_id, progress, f"Processing forged images... {i+1}/{len(forged_data)}")
        
        # Apply data augmentation
        job_queue.update_job_progress(job.job_id, 35.0, "Applying data augmentation...")
        augmenter = SignatureAugmentation(
            rotation_range=15.0,
            scale_range=(0.9, 1.1),
            brightness_range=0.2,
            blur_probability=0.3,
            thickness_variation=0.1
        )
        
        genuine_augmented, genuine_labels = augmenter.augment_batch(
            genuine_images, [True] * len(genuine_images), augmentation_factor=3
        )
        
        forged_augmented, forged_labels = augmenter.augment_batch(
            forged_images, [False] * len(forged_images), augmentation_factor=1
        )
        
        all_images = genuine_augmented + forged_augmented
        all_labels = genuine_labels + forged_labels
        
        job_queue.update_job_progress(job.job_id, 40.0, "Creating model record...")
        
        # Create model record in database
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
        job_queue.update_job_progress(job.job_id, 45.0, "Initializing AI model...")
        model_manager = SignatureVerificationModel()
        
        job_queue.update_job_progress(job.job_id, 50.0, "Training AI model...")
        t0 = time.time()
        history = model_manager.train_with_augmented_data(all_images, all_labels)
        
        # FIXED: Use correct method name
        job_queue.update_job_progress(job.job_id, 80.0, "Computing prototype and threshold...")
        centroid, threshold = model_manager.compute_centroid_and_adaptive_threshold(
            genuine_images,
            forged_images if len(forged_images) > 0 else None
        )
        
        # Calibrate threshold via EER (using the returned threshold)
        gen_emb = model_manager.embed_images(genuine_images)
        forg_emb = model_manager.embed_images(forged_images) if len(forged_images) else np.zeros((0, gen_emb.shape[1]))
        centroid_vec = np.array(centroid)
        gen_d = np.linalg.norm(gen_emb - centroid_vec, axis=1)
        forg_d = np.linalg.norm(forg_emb - centroid_vec, axis=1) if forg_emb.size else np.array([])
        
        # Calculate FAR and FRR at the threshold
        far = np.sum(forg_d <= threshold) / len(forg_d) if len(forg_d) > 0 else 0.0
        frr = np.sum(gen_d > threshold) / len(gen_d) if len(gen_d) > 0 else 0.0
        
        # Save models
        job_queue.update_job_progress(job.job_id, 85.0, "Saving model...")
        local_model_path = os.path.join(settings.LOCAL_MODELS_DIR, f"{model_uuid}.keras")
        local_embed_path = os.path.join(settings.LOCAL_MODELS_DIR, f"{model_uuid}_embed.keras")
        
        model_manager.save_model(local_model_path)
        model_manager.save_embedding_model(local_embed_path)
        
        # Upload to Supabase
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
                "far": float(far),
                "frr": float(frr)
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
            "accuracy": float(history.history.get("accuracy", [0])[-1]),
            "val_accuracy": float(val_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "train_time_s": float(train_time),
            "threshold": float(threshold),
            "far": float(far),
            "frr": float(frr),
            "model_id": numeric_model_id
        }
        
        job_queue.complete_job(job.job_id, result)
        
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
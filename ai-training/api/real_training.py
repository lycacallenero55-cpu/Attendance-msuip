# training.py - REAL AI TRAINING SYSTEM
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
from models.real_signature_model import RealSignatureVerificationModel
from utils.image_processing import preprocess_image
from utils.storage import save_to_supabase, cleanup_local_file
from utils.augmentation import SignatureAugmentation
from utils.job_queue import job_queue
from utils.training_callback import RealTimeMetricsCallback
from services.model_versioning import model_versioning_service
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global REAL AI model instance - can handle up to 150 students
real_ai_manager = RealSignatureVerificationModel(max_students=150)

async def train_signature_model(student, genuine_data, forged_data, job=None):
    """
    REAL AI training function - trains individual signature recognition
    
    This function trains the REAL AI system that actually learns to:
    1. Recognize individual students from their signatures
    2. Detect if signatures are genuine or forged
    """
    try:
        logger.info(f"🚀 Starting REAL AI training for student {student.id}")
        
        # Process and validate images
        genuine_images = []
        forged_images = []
        
        # Process genuine images
        for i, data in enumerate(genuine_data):
            image = Image.open(io.BytesIO(data))
            genuine_images.append(image)
            
            # Update progress if job provided
            if job:
                progress = 5.0 + (i + 1) / len(genuine_data) * 15.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing genuine images... {i+1}/{len(genuine_data)}")
        
        # Process forged images
        for i, data in enumerate(forged_data):
            image = Image.open(io.BytesIO(data))
            forged_images.append(image)
            
            # Update progress if job provided
            if job:
                progress = 20.0 + (i + 1) / len(forged_data) * 15.0
                job_queue.update_job_progress(job.job_id, progress, f"Processing forged images... {i+1}/{len(forged_data)}")
        
        if job:
            job_queue.update_job_progress(job.job_id, 35.0, "Preparing REAL AI training data...")
        
        # Prepare training data for REAL AI system
        training_data = {
            f"student_{student.id}": {
                'genuine': genuine_images,
                'forged': forged_images
            }
        }
        
        if job:
            job_queue.update_job_progress(job.job_id, 40.0, "Training REAL AI student recognition model...")
        
        # Train the REAL AI system
        training_result = real_ai_manager.train_real_ai_system(training_data)
        
        if job:
            job_queue.update_job_progress(job.job_id, 80.0, "Saving REAL AI models...")
        
        # Save models
        model_id = str(uuid.uuid4())
        base_path = f"models/real_ai_model_{model_id}"
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Save the REAL AI models
        real_ai_manager.save_models(base_path)
        
        # Upload to Supabase
        await save_to_supabase(f"{base_path}_student_model.keras", f"models/real_ai_student_model_{model_id}.keras")
        await save_to_supabase(f"{base_path}_authenticity_model.keras", f"models/real_ai_authenticity_model_{model_id}.keras")
        await save_to_supabase(f"{base_path}_student_mappings.json", f"models/real_ai_mappings_{model_id}.json")
        
        # Clean up local files
        cleanup_local_file(f"{base_path}_student_model.keras")
        cleanup_local_file(f"{base_path}_authenticity_model.keras")
        cleanup_local_file(f"{base_path}_student_mappings.json")
        
        # Save to database with REAL AI metadata
        await db_manager.save_trained_model(
            model_id=model_id,
            student_id=student.id,
            model_path=f"models/real_ai_student_model_{model_id}.keras",
            embedding_model_path=f"models/real_ai_authenticity_model_{model_id}.keras",
            prototype_centroid=None,  # Not used in REAL AI system
            prototype_threshold=None,  # Not used in REAL AI system
            training_metrics={
                'model_type': 'real_ai_individual_recognition',
                'student_recognition_accuracy': float(training_result['student_history']['accuracy'][-1]),
                'student_recognition_loss': float(training_result['student_history']['loss'][-1]),
                'authenticity_accuracy': float(training_result['authenticity_history']['accuracy'][-1]),
                'authenticity_loss': float(training_result['authenticity_history']['loss'][-1]),
                'epochs_trained': len(training_result['student_history']['accuracy']),
                'max_students': real_ai_manager.max_students
            }
        )
        
        if job:
            job_queue.update_job_progress(job.job_id, 100.0, "REAL AI training completed successfully!")
        
        logger.info(f"✅ REAL AI training completed for student {student.id}")
        
        return {
            "model_id": model_id,
            "student_id": student.id,
            "model_type": "real_ai_individual_recognition",
            "training_metrics": {
                'student_recognition_accuracy': float(training_result['student_history']['accuracy'][-1]),
                'student_recognition_loss': float(training_result['student_history']['loss'][-1]),
                'authenticity_accuracy': float(training_result['authenticity_history']['accuracy'][-1]),
                'authenticity_loss': float(training_result['authenticity_history']['loss'][-1]),
                'epochs_trained': len(training_result['student_history']['accuracy'])
            }
        }
        
    except Exception as e:
        logger.error(f"REAL AI training failed: {e}")
        if job:
            job_queue.update_job_progress(job.job_id, 0.0, f"REAL AI training failed: {str(e)}")
        raise

async def run_async_training(student_id: int, genuine_files: List[UploadFile], forged_files: List[UploadFile]):
    """
    Run REAL AI training asynchronously
    """
    try:
        # Create job for tracking
        job_id = str(uuid.uuid4())
        job_queue.create_job(job_id, "REAL AI Training", 0.0, "Starting REAL AI training...")
        
        # Read file contents before spawning background task
        genuine_data = []
        forged_data = []
        
        for file in genuine_files:
            content = await file.read()
            genuine_data.append(content)
        
        for file in forged_files:
            content = await file.read()
            forged_data.append(content)
        
        # Get student info
        student = await db_manager.get_student_by_id(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Start background training task
        asyncio.create_task(
            train_signature_model(student, genuine_data, forged_data, job_queue.get_job(job_id))
        )
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to start REAL AI training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.post("/start-async")
async def start_async_training(
    student_id: int = Form(...),
    genuine_files: List[UploadFile] = File(...),
    forged_files: List[UploadFile] = File(...)
):
    """
    Start REAL AI training asynchronously
    """
    try:
        logger.info(f"🚀 Starting REAL AI async training for student {student_id}")
        
        # Validate inputs
        if not genuine_files or not forged_files:
            raise HTTPException(status_code=400, detail="Both genuine and forged files are required")
        
        if len(genuine_files) < 5 or len(forged_files) < 5:
            raise HTTPException(status_code=400, detail="At least 5 genuine and 5 forged signatures are required")
        
        # Start training
        result = await run_async_training(student_id, genuine_files, forged_files)
        
        logger.info(f"✅ REAL AI async training started with job ID: {result['job_id']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REAL AI async training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/start")
async def start_training(
    student_id: int = Form(...),
    genuine_files: List[UploadFile] = File(...),
    forged_files: List[UploadFile] = File(...)
):
    """
    Start REAL AI training synchronously (for testing)
    """
    try:
        logger.info(f"🚀 Starting REAL AI sync training for student {student_id}")
        
        # Validate inputs
        if not genuine_files or not forged_files:
            raise HTTPException(status_code=400, detail="Both genuine and forged files are required")
        
        if len(genuine_files) < 5 or len(forged_files) < 5:
            raise HTTPException(status_code=400, detail="At least 5 genuine and 5 forged signatures are required")
        
        # Read file contents
        genuine_data = []
        forged_data = []
        
        for file in genuine_files:
            content = await file.read()
            genuine_data.append(content)
        
        for file in forged_files:
            content = await file.read()
            forged_data.append(content)
        
        # Get student info
        student = await db_manager.get_student_by_id(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Train the REAL AI system
        result = await train_signature_model(student, genuine_data, forged_data)
        
        logger.info(f"✅ REAL AI sync training completed: {result}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REAL AI sync training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/progress/{job_id}")
async def get_training_progress(job_id: str):
    """
    Get REAL AI training progress
    """
    try:
        job = job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get REAL AI training progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

@router.get("/models")
async def get_trained_models():
    """
    Get all trained REAL AI models
    """
    try:
        models = await db_manager.get_trained_models()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Failed to get REAL AI models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a REAL AI model
    """
    try:
        success = await db_manager.delete_trained_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"message": "REAL AI model deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete REAL AI model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")
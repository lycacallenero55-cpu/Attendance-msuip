from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import numpy as np
from PIL import Image
import io
import logging
import time

from models.database import db_manager
import tensorflow as tf
from tensorflow import keras
from models.signature_model import SignatureVerificationModel
from utils.image_processing import preprocess_image
from utils.storage import download_from_supabase
from utils.antispoofing import AntiSpoofingDetector
from services.model_versioning import model_versioning_service
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/verify")
async def verify_signature(
    student_id: str = Form(...),
    test_file: UploadFile = File(...)
):
    """Verify a signature for a student using stored prototype (no references required)"""
    
    try:
        # Resolve student (accept school_id or numeric id)
        student = await db_manager.get_student_by_school_id(student_id)
        if not student:
            try:
                numeric_id = int(student_id)
                student = await db_manager.get_student(numeric_id)
            except Exception:
                student = None
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        # Get trained model for this student
        models = await db_manager.get_trained_models(int(student["id"]))
        # Prefer newest completed model that has prototype metadata
        candidates = [m for m in models if m.get("status") == "completed" and m.get("prototype_centroid") is not None]
        if not candidates:
            # Fallback: any completed model
            candidates = [m for m in models if m.get("status") == "completed"]
        # Sort by created_at/updated_at desc if present
        def ts(m):
            return m.get("updated_at") or m.get("created_at") or ""
        candidates = sorted(candidates, key=ts, reverse=True)
        model = candidates[0] if candidates else None
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found for student")

        
        # Process test image
        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))
        test_processed = preprocess_image(test_image, settings.MODEL_IMAGE_SIZE)
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Perform anti-spoofing analysis
        spoofing_detector = AntiSpoofingDetector()
        spoofing_analysis = spoofing_detector.analyze_signature(test_processed)
        
        # Log spoofing analysis for monitoring
        logger.info(f"Anti-spoofing analysis for student {student_id}: {spoofing_analysis}")
        
        # Load the trained model for embedding
        model_manager = SignatureVerificationModel()
        # Prefer embedding-only artifact if present (no Lambda)
        embed_path_remote = model.get("embedding_model_path")
        if embed_path_remote:
            embed_local = await download_from_supabase(embed_path_remote)
            model_manager.embedding_model = keras.models.load_model(embed_local, safe_mode=False)
        else:
            model_path = await download_from_supabase(model["model_path"])
            try:
                model_manager.load_model(model_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Model artifact is from an old version. Please retrain this student and try again.")

        # Compute embedding for test image and compare to stored prototype
        test_arr = np.array(test_processed)
        test_tensor = tf.convert_to_tensor(test_arr)
        test_tensor = model_manager.preprocess_image(test_tensor)
        test_tensor = tf.expand_dims(test_tensor, axis=0)
        test_embedding = model_manager.embedding_model.predict(test_tensor, verbose=0)[0]

        centroid = np.array(model.get("prototype_centroid") or [])
        # Default threshold fallback
        try:
            threshold = float(model.get("prototype_threshold")) if model.get("prototype_threshold") is not None else 0.7
        except Exception:
            threshold = 0.7
        if centroid.size == 0:
            raise HTTPException(status_code=400, detail="Prototype not available for this model")

        dist = float(np.linalg.norm(test_embedding - centroid))
        is_genuine = dist <= threshold
        denom = threshold if threshold and threshold > 1e-6 else 1.0
        raw_score = 1.0 - dist / denom
        score = float(max(0.0, min(1.0, raw_score)))

        # Generate spoofing warning message
        spoofing_warning = spoofing_detector.get_spoofing_warning_message(spoofing_analysis)
        
        # Prepare verification result
        verification_result = {
            "success": True,
            "student_id": student_id,
            "match": is_genuine,
            "distance": dist,
            "threshold": threshold,
            "score": score,
            "message": "Verified against prototype",
            "antispoofing": {
                "is_potentially_spoofed": spoofing_analysis.get("is_potentially_spoofed", False),
                "is_likely_printed": spoofing_analysis.get("is_likely_printed", False),
                "is_low_quality": spoofing_analysis.get("is_low_quality", False),
                "printed_confidence": spoofing_analysis.get("printed_confidence", 0.0),
                "quality_score": spoofing_analysis.get("quality_score", 0.0),
                "warning_message": spoofing_warning
            }
        }
        
        # Record verification result for A/B testing and analytics
        try:
            processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            await model_versioning_service.record_verification_result(
                student_id=int(student["id"]),
                model_id=model["id"],
                verification_result=verification_result,
                processing_time_ms=processing_time
            )
        except Exception as e:
            logger.error(f"Error recording verification result: {e}")
            # Don't fail verification for analytics errors
        
        return verification_result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in signature verification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/models/{student_id}")
async def get_verification_models(student_id: int):
    """Get available models for verification for a specific student"""
    try:
        models = await db_manager.get_trained_models(student_id)
        completed_models = [m for m in models if m["status"] == "completed"]
        
        return {
            "student_id": student_id,
            "available_models": completed_models
        }
    
    except Exception as e:
        logger.error(f"Error getting verification models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
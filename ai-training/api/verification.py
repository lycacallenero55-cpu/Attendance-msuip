from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import numpy as np
from PIL import Image
import io
import logging
import time
import os

from models.database import db_manager
import tensorflow as tf
from tensorflow import keras
from models.signature_model import SignatureVerificationModel
from utils.image_processing import validate_image, preprocess_image
from utils.storage import download_from_supabase
from utils.antispoofing import AntiSpoofingDetector
from services.model_versioning import model_versioning_service
from services.model_cache import model_cache
from config import settings

async def load_signature_model(model):
    """
    Unified model loading helper for both verify and identify endpoints
    """
    model_manager = SignatureVerificationModel()
    
    # Try to load embedding-only model first (lighter/faster)
    embedding_path = model.get("embedding_model_path")
    if embedding_path:
        try:
            logger.info(f"🔄 Loading embedding model from: {embedding_path}")
            # Download from Supabase if needed
            if embedding_path.startswith("models/"):
                local_path = os.path.join(settings.LOCAL_MODELS_DIR, os.path.basename(embedding_path))
                if not os.path.exists(local_path):
                    # Download from Supabase
                    from utils.storage import download_from_supabase
                    temp_path = await download_from_supabase(embedding_path)
                    # Copy temp file to local path
                    import shutil
                    shutil.copy2(temp_path, local_path)
                    # Clean up temp file
                    from utils.storage import cleanup_local_file
                    cleanup_local_file(temp_path)
                embedding_path = local_path
            
            model_manager.embedding_model = keras.models.load_model(embedding_path)
            logger.info("✅ Embedding model loaded successfully")
            logger.info(f"Embedding model input shape: {model_manager.embedding_model.input_shape}")
            logger.info(f"Embedding model output shape: {model_manager.embedding_model.output_shape}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            embedding_path = None
    
    # Fallback to full model if embedding model failed
    if not embedding_path or not hasattr(model_manager, 'embedding_model'):
        try:
            model_path = model.get("model_path")
            if model_path.startswith("models/"):
                local_path = os.path.join(settings.LOCAL_MODELS_DIR, os.path.basename(model_path))
                if not os.path.exists(local_path):
                    # Download from Supabase
                    from utils.storage import download_from_supabase
                    temp_path = await download_from_supabase(model_path)
                    # Copy temp file to local path
                    import shutil
                    shutil.copy2(temp_path, local_path)
                    # Clean up temp file
                    from utils.storage import cleanup_local_file
                    cleanup_local_file(temp_path)
                model_path = local_path
            
            logger.info(f"🔄 Loading full model from: {model_path}")
            model_manager.model = keras.models.load_model(model_path)
            model_manager.embedding_model = model_manager.model.get_layer('embedding_model')
            logger.info("✅ Full model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=400, detail="Model artifact is from an old version. Please retrain this student and try again.")
    
    return model_manager

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

        # Validate test image
        if not validate_image(test_file):
            raise HTTPException(status_code=400, detail="Invalid test image")
        
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
        
        # Load the trained model for embedding using cache
        logger.info(f"Loading model for student {student_id}, model ID: {model.get('id')}")
        model_manager = SignatureVerificationModel()
        
        # Use cached model loading
        model_id = str(model.get('id'))
        model_path = model.get("model_path")
        embedding_path = model.get("embedding_model_path")
        
        cached_model = await model_cache.get_model(model_id, model_path, embedding_path)
        if cached_model:
            model_manager.embedding_model = cached_model
            logger.info("✅ Model loaded from cache successfully")
            logger.info(f"Model input shape: {cached_model.input_shape}")
            logger.info(f"Model output shape: {cached_model.output_shape}")
        else:
            raise HTTPException(status_code=400, detail="Model artifact is from an old version. Please retrain this student and try again.")

        # Compute embedding with light test-time augmentation (average of few variants)
        # FIXED: Use the same preprocessing pipeline as training
        logger.info("🔍 Computing embeddings using trained model...")
        embeddings = []
        
        # Test the model with a single image first
        test_embedding = model_manager.embed_images([test_processed])[0]
        logger.info(f"✅ Model inference successful! Embedding shape: {test_embedding.shape}")
        logger.info(f"Embedding range: [{test_embedding.min():.4f}, {test_embedding.max():.4f}]")
        logger.info(f"Embedding mean: {test_embedding.mean():.4f}, std: {test_embedding.std():.4f}")
        
        # Use single embedding for faster verification (TTA disabled for speed)
        test_embedding = test_embedding
        logger.info(f"✅ Final averaged embedding shape: {test_embedding.shape}")
        logger.info(f"Final embedding range: [{test_embedding.min():.4f}, {test_embedding.max():.4f}]")

        centroid = np.array(model.get("prototype_centroid") or [])
        logger.info(f"📊 Prototype centroid shape: {centroid.shape}")
        logger.info(f"Centroid range: [{centroid.min():.4f}, {centroid.max():.4f}]")
        
        # Default threshold fallback
        try:
            threshold = float(model.get("prototype_threshold")) if model.get("prototype_threshold") is not None else 0.7
        except Exception:
            threshold = 0.7
        
        if centroid.size == 0:
            raise HTTPException(status_code=400, detail="Prototype not available for this model")

        logger.info("🔍 Computing distance between test embedding and prototype centroid...")
        dist = float(np.linalg.norm(test_embedding - centroid))
        is_genuine = dist <= threshold
        # Calculate similarity score using sigmoid function for better calibration
        # Score approaches 1.0 when distance is much smaller than threshold
        # Score approaches 0.0 when distance is much larger than threshold
        import math
        # Use sigmoid: score = 1 / (1 + exp(k * (distance - threshold)))
        # k controls steepness, threshold is the decision boundary
        k = 5.0  # Steepness parameter
        score = 1.0 / (1.0 + math.exp(k * (dist - threshold)))
        score = float(max(0.0, min(1.0, score)))
        
        # Log the values for debugging
        logger.info(f"🎯 VERIFICATION RESULT:")
        logger.info(f"   Distance: {dist:.4f}")
        logger.info(f"   Threshold: {threshold:.4f}")
        logger.info(f"   Is genuine: {is_genuine}")
        logger.info(f"   Score: {score:.4f}")
        logger.info(f"   Test embedding shape: {test_embedding.shape}")
        logger.info(f"   Centroid shape: {centroid.shape}")

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

@router.post("/identify")
async def identify_signature_owner(
    test_file: UploadFile = File(...)
):
    """Identify the most likely student owner of a signature by scanning all saved models.

    This endpoint does NOT require selecting a student. It searches across all completed models
    using each model's own embedding network and stored prototype centroid/threshold.
    """
    try:
        # Validate image
        if not validate_image(test_file):
            raise HTTPException(status_code=400, detail="Invalid test image")

        # Load and preprocess once to PIL; will convert to tensors per model
        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))
        test_processed = preprocess_image(test_image, settings.MODEL_IMAGE_SIZE)

        # Fetch all completed models that have prototype metadata
        all_models = await db_manager.get_trained_models()
        candidates = [m for m in (all_models or []) if m.get("status") == "completed" and m.get("prototype_centroid") is not None]
        
        # Additional validation: ensure models have reasonable thresholds
        valid_candidates = []
        for model in candidates:
            threshold = model.get("prototype_threshold")
            if threshold is not None:
                try:
                    threshold_val = float(threshold)
                    if threshold_val >= 0.1:  # Reasonable minimum threshold
                        valid_candidates.append(model)
                    else:
                        logger.warning(f"Model {model.get('id')} has suspiciously low threshold: {threshold_val}")
                except:
                    logger.warning(f"Model {model.get('id')} has invalid threshold: {threshold}")
            else:
                logger.warning(f"Model {model.get('id')} has no threshold")
        
        candidates = valid_candidates
        if not candidates:
            raise HTTPException(status_code=404, detail="No valid trained models available")

        best = None
        best_score = -1.0
        best_is_match = False
        best_distance = None
        best_threshold = None
        best_model = None

        # For each model, load its embedding model and compute score against its centroid (with light TTA)
        logger.info(f"🔍 Testing against {len(candidates)} trained models...")
        logger.info(f"🔍 Candidate models: {[{'id': m.get('id'), 'student_id': m.get('student_id'), 'status': m.get('status')} for m in candidates]}")
        
        for i, model in enumerate(candidates):
            logger.info(f"Testing model {i+1}/{len(candidates)}: ID {model.get('id')}, Student ID {model.get('student_id')}")
            try:
                model_manager = SignatureVerificationModel()
                
                # Use cached model loading
                model_id = str(model.get('id'))
                model_path = model.get("model_path")
                embedding_path = model.get("embedding_model_path")
                
                logger.info(f"🔍 Loading model {model_id}: path={model_path}, embedding_path={embedding_path}")
                cached_model = await model_cache.get_model(model_id, model_path, embedding_path)
                if cached_model:
                    model_manager.embedding_model = cached_model
                    logger.info(f"✅ Model {model.get('id')} loaded from cache")
                else:
                    logger.warning(f"❌ Could not load model {model.get('id')}")
                    continue

                # Embed without TTA for faster identification
                logger.info(f"🔍 Computing embedding for model {model.get('id')}...")
                test_embedding = model_manager.embed_images([test_processed])[0]
                logger.info(f"✅ Embedding computed for model {model.get('id')}, shape: {test_embedding.shape}")

                centroid = np.array(model.get("prototype_centroid") or [])
                if centroid.size == 0:
                    continue
                try:
                    threshold = float(model.get("prototype_threshold")) if model.get("prototype_threshold") is not None else 0.7
                    # Ensure threshold is reasonable (not too low)
                    threshold = max(threshold, 0.3)  # Minimum threshold of 0.3
                except Exception:
                    threshold = 0.7

                dist = float(np.linalg.norm(test_embedding - centroid))
                # Use same sigmoid scoring as verify endpoint
                import math
                k = 5.0  # Steepness parameter
                score = 1.0 / (1.0 + math.exp(k * (dist - threshold)))
                score = float(max(0.0, min(1.0, score)))
                is_match = dist <= threshold

                logger.info(f"📊 Model {model.get('id')} (Student {model.get('student_id')}): dist={dist:.4f}, threshold={threshold:.4f}, score={score:.4f}, is_match={is_match}")

                if score > best_score:
                    logger.info(f"🏆 New best model: {model.get('id')} (Student {model.get('student_id')}) with score {score:.4f}")
                    best_score = score
                    best = {
                        "distance": dist,
                        "threshold": threshold,
                        "score": score,
                        "is_match": is_match,
                    }
                    best_is_match = is_match
                    best_distance = dist
                    best_threshold = threshold
                    best_model = model
            except Exception as e:
                logger.error(f"Identify: failed on model {model.get('id')}: {e}")
                continue

        if best is None or best_model is None:
            raise HTTPException(status_code=404, detail="Could not identify signature owner")

        logger.info(f"🎯 Final result: Best model {best_model.get('id')} (Student {best_model.get('student_id')}) with score {best_score:.4f}")

        # Lookup student info
        student = await db_manager.get_student(int(best_model["student_id"]))
        student_info = None
        if student:
            student_info = {
                "id": student["id"],
                "student_id": student.get("student_id"),
                "firstname": student.get("firstname"),
                "surname": student.get("surname"),
            }

        return {
            "success": True,
            "match": best_is_match,
            "predicted_student_id": student.get("id") if student else None,
            "predicted_student": student_info,
            "score": float(best_score),
            "distance": float(best_distance) if best_distance is not None else None,
            "threshold": float(best_threshold) if best_threshold is not None else None,
            "message": "Identified most likely owner from trained models",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in signature identification: {e}")
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

@router.get("/cache/stats")
async def get_cache_stats():
    """Get model cache statistics"""
    try:
        stats = model_cache.get_cache_stats()
        return {
            "success": True,
            "cache_stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/cache/clear")
async def clear_model_cache():
    """Clear the model cache"""
    try:
        await model_cache.clear_cache()
        return {
            "success": True,
            "message": "Model cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
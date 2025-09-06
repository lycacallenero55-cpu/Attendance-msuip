# verification.py - REAL AI VERIFICATION SYSTEM
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import io
import logging
import asyncio

from models.database import db_manager
from models.real_signature_model import RealSignatureVerificationModel
from utils.image_processing import validate_image, preprocess_image
from utils.storage import load_model_from_supabase
from services.model_cache import model_cache
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Global REAL AI model instance
real_ai_manager = RealSignatureVerificationModel(max_students=150)

@router.post("/identify")
async def identify_signature_owner(
    test_file: UploadFile = File(...)
):
    """
    REAL AI signature identification - identifies who wrote the signature
    
    This endpoint uses the REAL AI system to:
    1. Identify which student wrote the signature
    2. Determine if the signature is genuine or forged
    3. Provide meaningful confidence scores
    """
    logger.info("🚀 STARTING REAL AI SIGNATURE IDENTIFICATION")
    logger.info(f"📁 Received file: {test_file.filename}, size: {test_file.size}")
    
    try:
        # Validate image
        if not validate_image(test_file):
            logger.error("❌ Invalid image file")
            raise HTTPException(status_code=400, detail="Invalid test image")

        logger.info("✅ Image validation passed")
        
        # Load and preprocess image
        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))
        
        logger.info(f"✅ Image loaded, size: {test_image.size}")
        
        # Get all trained REAL AI models
        all_models = await db_manager.get_trained_models()
        logger.info(f"📊 Found {len(all_models) if all_models else 0} total models in database")
        
        # Filter for REAL AI models
        real_ai_models = [m for m in (all_models or []) if m.get("status") == "completed" and m.get("training_metrics", {}).get("model_type") == "real_ai_individual_recognition"]
        logger.info(f"📊 Found {len(real_ai_models)} REAL AI models")
        
        if not real_ai_models:
            logger.error("❌ No REAL AI models available")
            raise HTTPException(status_code=404, detail="No trained REAL AI models available")
        
        # Load the most recent REAL AI model
        latest_model = max(real_ai_models, key=lambda x: x.get("created_at", ""))
        model_id = latest_model.get("id")
        
        logger.info(f"🔍 Using REAL AI model {model_id}")
        
        # Load the REAL AI models
        base_path = f"models/real_ai_model_{model_id}"
        
        # Load student model
        student_model_path = latest_model.get("model_path")
        authenticity_model_path = latest_model.get("embedding_model_path")
        
        logger.info(f"🔍 Loading REAL AI models:")
        logger.info(f"   - Student model: {student_model_path}")
        logger.info(f"   - Authenticity model: {authenticity_model_path}")
        
        # Load models from Supabase
        try:
            # Load student recognition model
            student_model = await load_model_from_supabase(student_model_path)
            real_ai_manager.student_model = student_model
            logger.info("✅ Student recognition model loaded")
            
            # Load authenticity detection model
            authenticity_model = await load_model_from_supabase(authenticity_model_path)
            real_ai_manager.authenticity_model = authenticity_model
            logger.info("✅ Authenticity detection model loaded")
            
            # Load student mappings
            mappings_path = f"models/real_ai_mappings_{model_id}.json"
            mappings_model = await db_manager.get_trained_model(model_id)
            if mappings_model and mappings_model.get("training_metrics"):
                # Extract student mappings from training metrics
                metrics = mappings_model.get("training_metrics", {})
                if "student_mappings" in metrics:
                    mappings = metrics["student_mappings"]
                    real_ai_manager.student_to_id = mappings.get("student_to_id", {})
                    real_ai_manager.id_to_student = {int(k): v for k, v in mappings.get("id_to_student", {}).items()}
                    logger.info(f"✅ Student mappings loaded: {len(real_ai_manager.student_to_id)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load REAL AI models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load REAL AI models: {str(e)}")
        
        # Perform REAL AI verification
        logger.info("🔍 Performing REAL AI signature verification...")
        verification_result = real_ai_manager.verify_signature(test_image)
        
        logger.info(f"🎯 REAL AI verification completed:")
        logger.info(f"   - Predicted student: {verification_result['predicted_student_name']}")
        logger.info(f"   - Student confidence: {verification_result['student_confidence']:.3f}")
        logger.info(f"   - Is genuine: {verification_result['is_genuine']}")
        logger.info(f"   - Authenticity score: {verification_result['authenticity_score']:.3f}")
        logger.info(f"   - Overall confidence: {verification_result['overall_confidence']:.3f}")
        logger.info(f"   - Is unknown: {verification_result['is_unknown']}")
        
        # Format response
        response = {
            "predicted_student": {
                "id": verification_result["predicted_student_id"],
                "name": verification_result["predicted_student_name"]
            },
            "is_match": verification_result["is_genuine"],
            "confidence": verification_result["overall_confidence"],
            "student_confidence": verification_result["student_confidence"],
            "authenticity_score": verification_result["authenticity_score"],
            "is_unknown": verification_result["is_unknown"],
            "model_type": "real_ai_individual_recognition",
            "verification_details": {
                "student_recognition": {
                    "predicted_id": verification_result["predicted_student_id"],
                    "predicted_name": verification_result["predicted_student_name"],
                    "confidence": verification_result["student_confidence"]
                },
                "authenticity_detection": {
                    "is_genuine": verification_result["is_genuine"],
                    "score": verification_result["authenticity_score"]
                },
                "overall_assessment": {
                    "confidence": verification_result["overall_confidence"],
                    "is_unknown": verification_result["is_unknown"]
                }
            }
        }
        
        logger.info("✅ REAL AI signature identification completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REAL AI signature identification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Signature identification failed: {str(e)}")

@router.post("/verify")
async def verify_signature(
    test_file: UploadFile = File(...),
    student_id: Optional[int] = None
):
    """
    REAL AI signature verification - verifies if signature matches a specific student
    
    This endpoint verifies if a signature belongs to a specific student using the REAL AI system
    """
    logger.info("🚀 STARTING REAL AI SIGNATURE VERIFICATION")
    logger.info(f"📁 Received file: {test_file.filename}, size: {test_file.size}")
    logger.info(f"👤 Target student ID: {student_id}")
    
    try:
        # Validate image
        if not validate_image(test_file):
            logger.error("❌ Invalid image file")
            raise HTTPException(status_code=400, detail="Invalid test image")

        logger.info("✅ Image validation passed")
        
        # Load and preprocess image
        test_data = await test_file.read()
        test_image = Image.open(io.BytesIO(test_data))
        
        logger.info(f"✅ Image loaded, size: {test_image.size}")
        
        # Get all trained REAL AI models
        all_models = await db_manager.get_trained_models()
        logger.info(f"📊 Found {len(all_models) if all_models else 0} total models in database")
        
        # Filter for REAL AI models
        real_ai_models = [m for m in (all_models or []) if m.get("status") == "completed" and m.get("training_metrics", {}).get("model_type") == "real_ai_individual_recognition"]
        logger.info(f"📊 Found {len(real_ai_models)} REAL AI models")
        
        if not real_ai_models:
            logger.error("❌ No REAL AI models available")
            raise HTTPException(status_code=404, detail="No trained REAL AI models available")
        
        # Load the most recent REAL AI model
        latest_model = max(real_ai_models, key=lambda x: x.get("created_at", ""))
        model_id = latest_model.get("id")
        
        logger.info(f"🔍 Using REAL AI model {model_id}")
        
        # Load the REAL AI models (same as identify endpoint)
        student_model_path = latest_model.get("model_path")
        authenticity_model_path = latest_model.get("embedding_model_path")
        
        try:
            # Load models from Supabase
            student_model = await load_model_from_supabase(student_model_path)
            real_ai_manager.student_model = student_model
            
            authenticity_model = await load_model_from_supabase(authenticity_model_path)
            real_ai_manager.authenticity_model = authenticity_model
            
            # Load student mappings
            mappings_model = await db_manager.get_trained_model(model_id)
            if mappings_model and mappings_model.get("training_metrics"):
                metrics = mappings_model.get("training_metrics", {})
                if "student_mappings" in metrics:
                    mappings = metrics["student_mappings"]
                    real_ai_manager.student_to_id = mappings.get("student_to_id", {})
                    real_ai_manager.id_to_student = {int(k): v for k, v in mappings.get("id_to_student", {}).items()}
            
            logger.info("✅ REAL AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load REAL AI models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load REAL AI models: {str(e)}")
        
        # Perform REAL AI verification
        logger.info("🔍 Performing REAL AI signature verification...")
        verification_result = real_ai_manager.verify_signature(test_image)
        
        # Check if the predicted student matches the target student
        predicted_student_id = verification_result["predicted_student_id"]
        is_correct_student = (student_id is None) or (predicted_student_id == student_id)
        
        # Determine if it's a match based on both student identity and authenticity
        is_match = is_correct_student and verification_result["is_genuine"]
        
        logger.info(f"🎯 REAL AI verification completed:")
        logger.info(f"   - Predicted student: {verification_result['predicted_student_name']} (ID: {predicted_student_id})")
        logger.info(f"   - Target student: {student_id}")
        logger.info(f"   - Correct student: {is_correct_student}")
        logger.info(f"   - Is genuine: {verification_result['is_genuine']}")
        logger.info(f"   - Final match: {is_match}")
        logger.info(f"   - Overall confidence: {verification_result['overall_confidence']:.3f}")
        
        # Format response
        response = {
            "is_match": is_match,
            "confidence": verification_result["overall_confidence"],
            "student_confidence": verification_result["student_confidence"],
            "authenticity_score": verification_result["authenticity_score"],
            "predicted_student": {
                "id": verification_result["predicted_student_id"],
                "name": verification_result["predicted_student_name"]
            },
            "target_student_id": student_id,
            "is_correct_student": is_correct_student,
            "is_genuine": verification_result["is_genuine"],
            "is_unknown": verification_result["is_unknown"],
            "model_type": "real_ai_individual_recognition",
            "verification_details": {
                "student_recognition": {
                    "predicted_id": verification_result["predicted_student_id"],
                    "predicted_name": verification_result["predicted_student_name"],
                    "confidence": verification_result["student_confidence"],
                    "target_id": student_id,
                    "is_correct": is_correct_student
                },
                "authenticity_detection": {
                    "is_genuine": verification_result["is_genuine"],
                    "score": verification_result["authenticity_score"]
                },
                "overall_assessment": {
                    "is_match": is_match,
                    "confidence": verification_result["overall_confidence"],
                    "is_unknown": verification_result["is_unknown"]
                }
            }
        }
        
        logger.info("✅ REAL AI signature verification completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"REAL AI signature verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Signature verification failed: {str(e)}")

@router.get("/models")
async def get_verification_models():
    """
    Get all available REAL AI models for verification
    """
    try:
        models = await db_manager.get_trained_models()
        real_ai_models = [m for m in (models or []) if m.get("training_metrics", {}).get("model_type") == "real_ai_individual_recognition"]
        
        return {
            "models": real_ai_models,
            "total_count": len(real_ai_models),
            "model_type": "real_ai_individual_recognition"
        }
        
    except Exception as e:
        logger.error(f"Failed to get REAL AI verification models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")
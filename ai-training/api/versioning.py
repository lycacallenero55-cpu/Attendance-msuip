from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging
from datetime import datetime

from services.model_versioning import model_versioning_service
from models.database import db_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/models/{student_id}/versions")
async def get_model_versions(student_id: int):
    """Get all model versions for a student."""
    try:
        # Get all models for the student
        models = await db_manager.get_trained_models(student_id)
        if not models:
            return {"versions": []}
        
        # Get versions for each model
        all_versions = []
        for model in models:
            versions = await model_versioning_service.get_model_versions(model["id"])
            all_versions.extend(versions)
        
        # Sort by creation date
        all_versions.sort(key=lambda v: v.created_at, reverse=True)
        
        return {"versions": [v.__dict__ for v in all_versions]}
        
    except Exception as e:
        logger.error(f"Error getting model versions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/models/{student_id}/active")
async def get_active_model(student_id: int):
    """Get the currently active model for a student."""
    try:
        active_model = await model_versioning_service.get_active_model(student_id)
        if not active_model:
            raise HTTPException(status_code=404, detail="No active model found")
        
        return {"model": active_model.__dict__}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/models/{model_id}/versions")
async def create_model_version(
    model_id: int,
    version_notes: Optional[str] = None,
    created_by: Optional[str] = None
):
    """Create a new version of a model."""
    try:
        # Get model performance metrics
        model = await db_manager.get_trained_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        performance_metrics = {
            "accuracy": model.get("accuracy"),
            "val_accuracy": model.get("val_accuracy"),
            "precision": model.get("precision"),
            "recall": model.get("recall"),
            "f1": model.get("f1"),
            "far": model.get("far"),
            "frr": model.get("frr"),
            "threshold": model.get("prototype_threshold")
        }
        
        version = await model_versioning_service.create_model_version(
            model_id=model_id,
            version_notes=version_notes,
            created_by=created_by,
            performance_metrics=performance_metrics
        )
        
        return {"version": version.__dict__}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model version: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/models/versions/{version_id}/activate")
async def activate_model_version(
    version_id: int,
    activated_by: Optional[str] = None
):
    """Activate a specific model version."""
    try:
        success = await model_versioning_service.activate_model_version(
            version_id=version_id,
            activated_by=activated_by
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Version not found or could not be activated")
        
        return {"message": "Model version activated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating model version: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/ab-tests")
async def create_ab_test(
    student_id: int,
    model_a_id: int,
    model_b_id: int,
    test_name: str,
    description: Optional[str] = None,
    traffic_split: float = 0.5,
    created_by: Optional[str] = None
):
    """Create an A/B test between two model versions."""
    try:
        ab_test = await model_versioning_service.create_ab_test(
            student_id=student_id,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            test_name=test_name,
            description=description,
            traffic_split=traffic_split,
            created_by=created_by
        )
        
        return {"ab_test": ab_test.__dict__}
        
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ab-tests/{ab_test_id}/results")
async def get_ab_test_results(ab_test_id: int):
    """Get A/B test results and statistics."""
    try:
        results = await model_versioning_service.get_ab_test_results(ab_test_id)
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error getting A/B test results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/models/{model_id}/audit-trail")
async def get_model_audit_trail(model_id: int):
    """Get audit trail for a model."""
    try:
        audit_trail = await model_versioning_service.get_model_audit_trail(model_id)
        return {"audit_trail": audit_trail}
        
    except Exception as e:
        logger.error(f"Error getting audit trail: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/students/{student_id}/history")
async def get_student_model_history(student_id: int):
    """Get complete model history for a student."""
    try:
        history = await model_versioning_service.get_student_model_history(student_id)
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error getting student model history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/verification-results")
async def record_verification_result(
    student_id: int,
    model_id: int,
    verification_result: dict,
    processing_time_ms: int,
    ab_test_id: Optional[int] = None
):
    """Record a verification result for analysis."""
    try:
        result_id = await model_versioning_service.record_verification_result(
            student_id=student_id,
            model_id=model_id,
            verification_result=verification_result,
            processing_time_ms=processing_time_ms,
            ab_test_id=ab_test_id
        )
        
        return {"result_id": result_id}
        
    except Exception as e:
        logger.error(f"Error recording verification result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
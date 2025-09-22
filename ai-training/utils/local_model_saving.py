"""
Local model saving utilities for signature models.
Provides functions to save models locally without S3 upload.
"""

import os
import json
import logging
from typing import Dict, List
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def save_signature_models_locally(
    local_manager, 
    model_type: str, 
    model_uuid: str
) -> List[Dict]:
    """
    Save signature models locally without S3 upload.
    
    Args:
        local_manager: Local model manager instance
        model_type: Type of model ('individual' or 'global')
        model_uuid: Unique identifier for the model
    
    Returns:
        List of saved file information
    """
    saved_files = []
    
    try:
        # Create local models directory
        models_dir = Path("local_models")
        models_dir.mkdir(exist_ok=True)
        
        if model_type == "individual":
            # Save individual student models
            if hasattr(local_manager, 'get_models'):
                models = local_manager.get_models()
                for student_id, model in models.items():
                    student_uuid = f"{model_uuid}_{student_id}"
                    file_path = models_dir / f"individual_{student_uuid}.keras"
                    
                    # Save the model
                    if hasattr(model, 'save'):
                        model.save(str(file_path), save_format='keras')
                    else:
                        import pickle
                        with open(file_path, 'wb') as f:
                            pickle.dump(model, f)
                    
                    saved_files.append({
                        "student_id": student_id,
                        "file_path": str(file_path),
                        "model_type": "individual",
                        "model_uuid": student_uuid
                    })
                    
                    logger.info(f"✅ Individual model for student {student_id} saved locally: {file_path}")
            else:
                # Fallback: save the main model
                main_model = getattr(local_manager, 'model', None)
                if main_model:
                    file_path = models_dir / f"individual_{model_uuid}.keras"
                    
                    if hasattr(main_model, 'save'):
                        main_model.save(str(file_path), save_format='keras')
                    else:
                        import pickle
                        with open(file_path, 'wb') as f:
                            pickle.dump(main_model, f)
                    
                    saved_files.append({
                        "file_path": str(file_path),
                        "model_type": "individual",
                        "model_uuid": model_uuid
                    })
                    
                    logger.info(f"✅ Individual model saved locally: {file_path}")
        
        elif model_type == "global":
            # Save global model
            global_model = getattr(local_manager, 'global_model', None)
            if global_model:
                file_path = models_dir / f"global_{model_uuid}.keras"
                
                if hasattr(global_model, 'save'):
                    global_model.save(str(file_path), save_format='keras')
                else:
                    import pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(global_model, f)
                
                saved_files.append({
                    "file_path": str(file_path),
                    "model_type": "global",
                    "model_uuid": model_uuid
                })
                
                logger.info(f"✅ Global model saved locally: {file_path}")
        
        # Save metadata
        metadata = {
            "model_type": model_type,
            "model_uuid": model_uuid,
            "saved_files": saved_files,
            "timestamp": str(uuid.uuid4())
        }
        
        metadata_path = models_dir / f"metadata_{model_uuid}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Local saving completed: {len(saved_files)} {model_type} models saved locally")
        return saved_files
        
    except Exception as e:
        logger.error(f"❌ Local model saving failed: {e}")
        raise


def cleanup_local_models(model_uuid: str = None):
    """
    Clean up local model files.
    
    Args:
        model_uuid: Optional specific model UUID to clean up
    """
    try:
        models_dir = Path("local_models")
        if not models_dir.exists():
            return
        
        if model_uuid:
            # Clean up specific model
            pattern = f"*{model_uuid}*"
            for file_path in models_dir.glob(pattern):
                file_path.unlink()
                logger.info(f"🗑️ Cleaned up local file: {file_path}")
        else:
            # Clean up all local models
            for file_path in models_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"🗑️ Cleaned up local file: {file_path}")
        
        logger.info("✅ Local model cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Local model cleanup failed: {e}")


def get_local_model_path(model_uuid: str, model_type: str) -> str:
    """
    Get the local file path for a saved model.
    
    Args:
        model_uuid: Model UUID
        model_type: Model type ('individual' or 'global')
    
    Returns:
        Local file path
    """
    models_dir = Path("local_models")
    return str(models_dir / f"{model_type}_{model_uuid}.keras")
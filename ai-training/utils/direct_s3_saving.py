"""
Direct S3 saving utilities for signature models.
Provides functions to save models directly to S3 without local file storage.
"""

import uuid
import io
import logging
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

from utils.s3_storage import upload_model_file, upload_bytes
from models.signature_embedding_model import SignatureEmbeddingModel
from models.global_signature_classifier import GlobalSignatureClassifier

logger = logging.getLogger(__name__)


class DirectS3ModelSaver:
    """Helper class for saving models directly to S3."""
    
    def __init__(self):
        self.uploaded_files = []
    
    def save_model(self, model, model_type: str, model_uuid: str) -> Tuple[str, str]:
        """Save a model directly to S3 and return (s3_key, s3_url)."""
        try:
            # Convert model to bytes
            model_bytes = self._model_to_bytes(model)
            
            # Upload to S3
            s3_key, s3_url = upload_model_file(
                model_bytes, 
                model_type, 
                model_uuid, 
                "keras"
            )
            
            self.uploaded_files.append({
                "type": model_type,
                "uuid": model_uuid,
                "s3_key": s3_key,
                "s3_url": s3_url
            })
            
            logger.info(f"✅ {model_type} model {model_uuid} saved to S3: {s3_key}")
            return s3_key, s3_url
            
        except Exception as e:
            logger.error(f"❌ Failed to save {model_type} model {model_uuid} to S3: {e}")
            raise
    
    def _model_to_bytes(self, model) -> bytes:
        """Convert a TensorFlow model to bytes."""
        buffer = io.BytesIO()
        
        if hasattr(model, 'save'):
            # For Keras models
            model.save(buffer, save_format='keras')
        else:
            # For other model types, try to serialize
            import pickle
            pickle.dump(model, buffer)
        
        buffer.seek(0)
        return buffer.getvalue()


def save_signature_models_directly(
    local_manager, 
    model_type: str, 
    model_uuid: str
) -> List[Dict]:
    """
    Save signature models directly to S3 using the local manager.
    
    Args:
        local_manager: Local model manager instance
        model_type: Type of model ('individual' or 'global')
        model_uuid: Unique identifier for the model
    
    Returns:
        List of uploaded file information
    """
    saver = DirectS3ModelSaver()
    uploaded_files = []
    
    try:
        if model_type == "individual":
            # Save individual student models
            if hasattr(local_manager, 'get_models'):
                models = local_manager.get_models()
                for student_id, model in models.items():
                    student_uuid = f"{model_uuid}_{student_id}"
                    s3_key, s3_url = saver.save_model(model, "individual", student_uuid)
                    uploaded_files.append({
                        "student_id": student_id,
                        "s3_key": s3_key,
                        "s3_url": s3_url,
                        "model_type": "individual"
                    })
            else:
                # Fallback: save the main model
                main_model = getattr(local_manager, 'model', None)
                if main_model:
                    s3_key, s3_url = saver.save_model(main_model, "individual", model_uuid)
                    uploaded_files.append({
                        "s3_key": s3_key,
                        "s3_url": s3_url,
                        "model_type": "individual"
                    })
        
        elif model_type == "global":
            # Save global model
            global_model = getattr(local_manager, 'global_model', None)
            if global_model:
                s3_key, s3_url = saver.save_model(global_model, "global", model_uuid)
                uploaded_files.append({
                    "s3_key": s3_key,
                    "s3_url": s3_url,
                    "model_type": "global"
                })
        
        logger.info(f"✅ Saved {len(uploaded_files)} {model_type} models directly to S3")
        return uploaded_files
        
    except Exception as e:
        logger.error(f"❌ Failed to save {model_type} models directly to S3: {e}")
        raise


def save_global_model_directly(
    global_classifier: GlobalSignatureClassifier, 
    model_type: str, 
    model_uuid: str
) -> Tuple[str, str]:
    """
    Save a global signature classifier directly to S3.
    
    Args:
        global_classifier: The trained global classifier
        model_type: Type of model ('global')
        model_uuid: Unique identifier for the model
    
    Returns:
        Tuple of (s3_key, s3_url)
    """
    saver = DirectS3ModelSaver()
    
    try:
        # Get the underlying Keras model
        keras_model = getattr(global_classifier, 'model', None)
        if not keras_model:
            raise ValueError("Global classifier does not have a Keras model")
        
        # Save the model
        s3_key, s3_url = saver.save_model(keras_model, model_type, model_uuid)
        
        logger.info(f"✅ Global model {model_uuid} saved directly to S3: {s3_key}")
        return s3_key, s3_url
        
    except Exception as e:
        logger.error(f"❌ Failed to save global model {model_uuid} directly to S3: {e}")
        raise


def save_individual_model_directly(
    embedding_model: SignatureEmbeddingModel, 
    student_id: str,
    model_type: str, 
    model_uuid: str
) -> Tuple[str, str]:
    """
    Save an individual signature embedding model directly to S3.
    
    Args:
        embedding_model: The trained embedding model
        student_id: Student identifier
        model_type: Type of model ('individual')
        model_uuid: Unique identifier for the model
    
    Returns:
        Tuple of (s3_key, s3_url)
    """
    saver = DirectS3ModelSaver()
    
    try:
        # Get the underlying Keras model
        keras_model = getattr(embedding_model, 'model', None)
        if not keras_model:
            raise ValueError("Embedding model does not have a Keras model")
        
        # Save the model with student-specific UUID
        student_uuid = f"{model_uuid}_{student_id}"
        s3_key, s3_url = saver.save_model(keras_model, model_type, student_uuid)
        
        logger.info(f"✅ Individual model for student {student_id} saved directly to S3: {s3_key}")
        return s3_key, s3_url
        
    except Exception as e:
        logger.error(f"❌ Failed to save individual model for student {student_id} directly to S3: {e}")
        raise
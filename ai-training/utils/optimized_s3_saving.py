"""
Optimized S3 saving utilities for signature models with parallel uploads.
"""

import asyncio
import logging
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import uuid

from utils.s3_storage import upload_model_file

logger = logging.getLogger(__name__)


def save_signature_models_optimized(
    local_manager, 
    model_type: str, 
    model_uuid: str,
    max_workers: int = 4
) -> List[Dict]:
    """
    Save signature models to S3 with optimized parallel uploads.
    
    Args:
        local_manager: Local model manager instance
        model_type: Type of model ('individual' or 'global')
        model_uuid: Unique identifier for the model
        max_workers: Maximum number of parallel upload threads
    
    Returns:
        List of uploaded file information
    """
    uploaded_files = []
    
    try:
        if model_type == "individual":
            # Get all individual models
            if hasattr(local_manager, 'get_models'):
                models = local_manager.get_models()
                
                # Prepare upload tasks
                upload_tasks = []
                for student_id, model in models.items():
                    student_uuid = f"{model_uuid}_{student_id}"
                    upload_tasks.append((model, "individual", student_uuid, student_id))
                
                # Execute parallel uploads
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for model, mtype, muuid, sid in upload_tasks:
                        future = executor.submit(_upload_model_worker, model, mtype, muuid, sid)
                        futures.append(future)
                    
                    # Collect results
                    for future in futures:
                        try:
                            result = future.result()
                            uploaded_files.append(result)
                        except Exception as e:
                            logger.error(f"Failed to upload model: {e}")
                            raise
            else:
                # Fallback: save the main model
                main_model = getattr(local_manager, 'model', None)
                if main_model:
                    result = _upload_model_worker(main_model, "individual", model_uuid, None)
                    uploaded_files.append(result)
        
        elif model_type == "global":
            # Save global model
            global_model = getattr(local_manager, 'global_model', None)
            if global_model:
                result = _upload_model_worker(global_model, "global", model_uuid, None)
                uploaded_files.append(result)
        
        logger.info(f"✅ Optimized upload completed: {len(uploaded_files)} {model_type} models saved to S3")
        return uploaded_files
        
    except Exception as e:
        logger.error(f"❌ Optimized S3 saving failed: {e}")
        raise


def _upload_model_worker(model, model_type: str, model_uuid: str, student_id: str = None) -> Dict:
    """
    Worker function for parallel model uploads.
    
    Args:
        model: The model to upload
        model_type: Type of model
        model_uuid: Unique identifier
        student_id: Optional student ID
    
    Returns:
        Upload result dictionary
    """
    try:
        # Convert model to bytes and upload
        import io
        buffer = io.BytesIO()
        
        if hasattr(model, 'save'):
            model.save(buffer, save_format='keras')
        else:
            import pickle
            pickle.dump(model, buffer)
        
        buffer.seek(0)
        model_bytes = buffer.getvalue()
        
        # Upload to S3
        s3_key, s3_url = upload_model_file(
            model_bytes, 
            model_type, 
            model_uuid, 
            "keras"
        )
        
        result = {
            "s3_key": s3_key,
            "s3_url": s3_url,
            "model_type": model_type,
            "model_uuid": model_uuid
        }
        
        if student_id:
            result["student_id"] = student_id
        
        logger.info(f"✅ {model_type} model {model_uuid} uploaded to S3: {s3_key}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to upload {model_type} model {model_uuid}: {e}")
        raise


async def save_signature_models_async(
    local_manager, 
    model_type: str, 
    model_uuid: str,
    max_workers: int = 4
) -> List[Dict]:
    """
    Async version of optimized S3 saving.
    
    Args:
        local_manager: Local model manager instance
        model_type: Type of model ('individual' or 'global')
        model_uuid: Unique identifier for the model
        max_workers: Maximum number of parallel upload threads
    
    Returns:
        List of uploaded file information
    """
    loop = asyncio.get_event_loop()
    
    # Run the synchronous optimized saving in a thread pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(
            executor,
            save_signature_models_optimized,
            local_manager,
            model_type,
            model_uuid,
            max_workers
        )
    
    return result
"""
Model caching service for fast verification
"""
import asyncio
import logging
from typing import Dict, Optional
import tensorflow as tf
from tensorflow import keras
import os
from config import settings
from utils.storage import download_from_supabase

logger = logging.getLogger(__name__)

class ModelCache:
    """In-memory model cache to avoid reloading models on every request"""
    
    def __init__(self):
        self._cache: Dict[str, keras.Model] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_metadata: Dict[str, dict] = {}
    
    async def get_model(self, model_id: str, model_path: str, embedding_path: Optional[str] = None) -> Optional[keras.Model]:
        """Get model from cache or load it if not cached"""
        async with self._cache_lock:
            # Check if model is already cached
            if model_id in self._cache:
                logger.info(f"✅ Using cached model {model_id}")
                return self._cache[model_id]
            
            # Load model and cache it
            try:
                model = await self._load_model(model_path, embedding_path)
                if model:
                    self._cache[model_id] = model
                    self._cache_metadata[model_id] = {
                        "model_path": model_path,
                        "embedding_path": embedding_path,
                        "loaded_at": asyncio.get_event_loop().time()
                    }
                    logger.info(f"✅ Loaded and cached model {model_id}")
                    return model
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return None
    
    async def _load_model(self, model_path: str, embedding_path: Optional[str] = None) -> Optional[keras.Model]:
        """Load model from storage"""
        try:
            # Try embedding model first (lighter/faster)
            if embedding_path:
                try:
                    local_path = await self._download_if_needed(embedding_path)
                    model = keras.models.load_model(local_path, safe_mode=False)
                    logger.info(f"✅ Loaded embedding model from {local_path}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load embedding model: {e}")
            
            # Fallback to full model
            local_path = await self._download_if_needed(model_path)
            model = keras.models.load_model(local_path, safe_mode=False)
            logger.info(f"✅ Loaded full model from {local_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    async def _download_if_needed(self, remote_path: str) -> str:
        """Download model from Supabase if not already local"""
        if remote_path.startswith("models/"):
            local_path = os.path.join(settings.LOCAL_MODELS_DIR, os.path.basename(remote_path))
            if not os.path.exists(local_path):
                logger.info(f"📥 Downloading model from {remote_path} to {local_path}")
                temp_path = await download_from_supabase(remote_path)
                # Copy temp file to local path
                import shutil
                shutil.copy2(temp_path, local_path)
                # Clean up temp file
                from utils.storage import cleanup_local_file
                cleanup_local_file(temp_path)
            return local_path
        return remote_path
    
    async def clear_cache(self):
        """Clear the model cache"""
        async with self._cache_lock:
            self._cache.clear()
            self._cache_metadata.clear()
            logger.info("🧹 Model cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cached_models": len(self._cache),
            "cache_metadata": self._cache_metadata
        }

# Global model cache instance
model_cache = ModelCache()
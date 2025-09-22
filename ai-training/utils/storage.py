"""
General storage utilities for file management.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_local_file(file_path: str) -> bool:
    """
    Clean up a local file if it exists.
    
    Args:
        file_path: Path to the file to clean up
    
    Returns:
        True if file was cleaned up, False if it didn't exist
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ Cleaned up local file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Failed to clean up local file {file_path}: {e}")
        return False


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create directory {directory_path}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size in bytes, or None if file doesn't exist
    """
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return None
    except Exception as e:
        logger.error(f"❌ Failed to get file size for {file_path}: {e}")
        return None


def is_file_readable(file_path: str) -> bool:
    """
    Check if a file is readable.
    
    Args:
        file_path: Path to the file
    
    Returns:
        True if file is readable, False otherwise
    """
    try:
        return os.path.exists(file_path) and os.access(file_path, os.R_OK)
    except Exception as e:
        logger.error(f"❌ Failed to check file readability for {file_path}: {e}")
        return False


def cleanup_temp_files(temp_dir: str = "temp", max_age_hours: int = 24):
    """
    Clean up temporary files older than specified age.
    
    Args:
        temp_dir: Directory containing temporary files
        max_age_hours: Maximum age in hours before cleanup
    """
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return
        
        cleaned_count = 0
        for file_path in temp_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.info(f"🗑️ Cleaned up old temp file: {file_path}")
        
        if cleaned_count > 0:
            logger.info(f"✅ Cleaned up {cleaned_count} old temporary files")
        
    except Exception as e:
        logger.error(f"❌ Failed to clean up temporary files: {e}")
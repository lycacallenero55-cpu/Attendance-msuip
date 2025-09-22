"""
Storage utility functions for file cleanup and management.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def cleanup_local_file(file_path: str) -> bool:
    """
    Clean up a local file by removing it from the filesystem.
    
    Args:
        file_path: Path to the file to be removed
        
    Returns:
        bool: True if file was successfully removed, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Successfully cleaned up file: {file_path}")
            return True
        else:
            logger.warning(f"File not found for cleanup: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")
        return False

def cleanup_local_files(file_paths: list) -> dict:
    """
    Clean up multiple local files.
    
    Args:
        file_paths: List of file paths to be removed
        
    Returns:
        dict: Dictionary with file paths as keys and cleanup status as values
    """
    results = {}
    for file_path in file_paths:
        results[file_path] = cleanup_local_file(file_path)
    return results

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def get_file_size(file_path: str) -> Optional[int]:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: File size in bytes, or None if file doesn't exist
    """
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return None
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return None

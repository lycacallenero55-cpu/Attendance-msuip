"""
Model versioning service for tracking model versions, A/B testing, and audit trails.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ModelVersioningService:
    """Service for managing model versions and A/B testing."""
    
    def __init__(self):
        self.versions = {}  # In-memory storage for demo purposes
        self.active_models = {}  # student_id -> active_model_id
        self.ab_tests = {}  # ab_test_id -> test_config
        self.audit_trail = []  # List of audit events
    
    async def get_model_versions(self, model_id: str) -> List[Dict]:
        """Get all versions for a model."""
        try:
            versions = self.versions.get(model_id, [])
            return versions
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_id}: {e}")
            return []
    
    async def get_active_model(self, student_id: str) -> Optional[Dict]:
        """Get the currently active model for a student."""
        try:
            active_model_id = self.active_models.get(student_id)
            if not active_model_id:
                return None
            
            # Find the model in versions
            for model_id, versions in self.versions.items():
                for version in versions:
                    if version.get("id") == active_model_id:
                        return version
            
            return None
        except Exception as e:
            logger.error(f"Failed to get active model for student {student_id}: {e}")
            return None
    
    async def create_model_version(
        self, 
        model_id: str, 
        version_notes: str = "",
        created_by: str = "system"
    ) -> Dict:
        """Create a new model version."""
        try:
            version_id = str(uuid.uuid4())
            version = {
                "id": version_id,
                "model_id": model_id,
                "version_notes": version_notes,
                "created_by": created_by,
                "created_at": datetime.now().isoformat(),
                "is_active": False
            }
            
            if model_id not in self.versions:
                self.versions[model_id] = []
            
            self.versions[model_id].append(version)
            
            # Add to audit trail
            self.audit_trail.append({
                "action": "version_created",
                "model_id": model_id,
                "version_id": version_id,
                "created_by": created_by,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Created model version {version_id} for model {model_id}")
            return version
            
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            raise
    
    async def activate_model_version(
        self, 
        version_id: str, 
        activated_by: str = "system"
    ) -> bool:
        """Activate a specific model version."""
        try:
            # Find the version
            for model_id, versions in self.versions.items():
                for version in versions:
                    if version.get("id") == version_id:
                        # Deactivate other versions of the same model
                        for v in versions:
                            v["is_active"] = False
                        
                        # Activate this version
                        version["is_active"] = True
                        
                        # Update active models
                        student_id = version.get("student_id", "unknown")
                        self.active_models[student_id] = version_id
                        
                        # Add to audit trail
                        self.audit_trail.append({
                            "action": "version_activated",
                            "model_id": model_id,
                            "version_id": version_id,
                            "activated_by": activated_by,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        logger.info(f"Activated model version {version_id}")
                        return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to activate model version {version_id}: {e}")
            return False
    
    async def create_ab_test(
        self, 
        student_id: str, 
        model_a_id: str, 
        model_b_id: str,
        test_name: str = "",
        created_by: str = "system"
    ) -> Dict:
        """Create an A/B test between two model versions."""
        try:
            ab_test_id = str(uuid.uuid4())
            ab_test = {
                "id": ab_test_id,
                "student_id": student_id,
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "test_name": test_name,
                "created_by": created_by,
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "results": {
                    "model_a_wins": 0,
                    "model_b_wins": 0,
                    "total_tests": 0
                }
            }
            
            self.ab_tests[ab_test_id] = ab_test
            
            # Add to audit trail
            self.audit_trail.append({
                "action": "ab_test_created",
                "ab_test_id": ab_test_id,
                "student_id": student_id,
                "created_by": created_by,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Created A/B test {ab_test_id} for student {student_id}")
            return ab_test
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            raise
    
    async def get_ab_test_results(self, ab_test_id: str) -> Optional[Dict]:
        """Get A/B test results and statistics."""
        try:
            ab_test = self.ab_tests.get(ab_test_id)
            if not ab_test:
                return None
            
            return {
                "ab_test": ab_test,
                "results": ab_test.get("results", {}),
                "status": ab_test.get("status", "unknown")
            }
        except Exception as e:
            logger.error(f"Failed to get A/B test results for {ab_test_id}: {e}")
            return None
    
    async def get_model_audit_trail(self, model_id: str) -> List[Dict]:
        """Get audit trail for a model."""
        try:
            model_audit = [
                event for event in self.audit_trail 
                if event.get("model_id") == model_id
            ]
            return model_audit
        except Exception as e:
            logger.error(f"Failed to get audit trail for model {model_id}: {e}")
            return []
    
    async def get_student_model_history(self, student_id: str) -> List[Dict]:
        """Get complete model history for a student."""
        try:
            history = []
            
            # Get all models for this student
            for model_id, versions in self.versions.items():
                for version in versions:
                    if version.get("student_id") == student_id:
                        history.append(version)
            
            # Sort by creation date
            history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return history
            
        except Exception as e:
            logger.error(f"Failed to get model history for student {student_id}: {e}")
            return []
    
    async def record_verification_result(
        self, 
        student_id: str, 
        model_id: str, 
        is_match: bool, 
        confidence: float,
        test_data: Dict = None
    ) -> str:
        """Record a verification result for analysis."""
        try:
            result_id = str(uuid.uuid4())
            result = {
                "id": result_id,
                "student_id": student_id,
                "model_id": model_id,
                "is_match": is_match,
                "confidence": confidence,
                "test_data": test_data or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to audit trail
            self.audit_trail.append({
                "action": "verification_recorded",
                "result_id": result_id,
                "student_id": student_id,
                "model_id": model_id,
                "is_match": is_match,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Recorded verification result {result_id} for student {student_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Failed to record verification result: {e}")
            raise


# Global instance
model_versioning_service = ModelVersioningService()
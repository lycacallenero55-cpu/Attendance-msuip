import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from models.database import db_manager

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Represents a model version with metadata."""
    id: int
    model_id: int
    version: int
    created_at: datetime
    created_by: Optional[str]
    version_notes: Optional[str]
    performance_metrics: Optional[Dict[str, Any]]
    model_artifacts: Optional[Dict[str, str]]
    is_active: bool

@dataclass
class ABTest:
    """Represents an A/B test between two model versions."""
    id: int
    student_id: int
    model_a_id: int
    model_b_id: int
    test_name: str
    description: Optional[str]
    start_date: datetime
    end_date: Optional[datetime]
    is_active: bool
    traffic_split: float
    results: Optional[Dict[str, Any]]
    created_by: Optional[str]

class ModelVersioningService:
    """Service for managing model versions, A/B testing, and audit trails."""
    
    def __init__(self):
        self.db = db_manager
    
    async def create_model_version(
        self, 
        model_id: int, 
        version_notes: str = None,
        created_by: str = None,
        performance_metrics: Dict[str, Any] = None
    ) -> ModelVersion:
        """Create a new version record for a model."""
        try:
            # Get current model info
            model = await self.db.get_trained_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Get next version number
            versions = await self.get_model_versions(model_id)
            next_version = max([v.version for v in versions], default=0) + 1
            
            # Create version record
            version_data = {
                "model_id": model_id,
                "version": next_version,
                "created_by": created_by,
                "version_notes": version_notes,
                "performance_metrics": performance_metrics,
                "model_artifacts": {
                    "model_path": model.get("model_path"),
                    "embedding_model_path": model.get("embedding_model_path")
                },
                "is_active": True
            }
            
            result = await self.db.create_model_version(version_data)
            
            # Deactivate previous versions for this student
            await self.deactivate_other_versions(model["student_id"], model_id)
            
            # Log the action
            await self.log_model_action(
                model_id, 
                "version_created", 
                new_values={"version": next_version, "notes": version_notes},
                performed_by=created_by
            )
            
            return ModelVersion(
                id=result["id"],
                model_id=model_id,
                version=next_version,
                created_at=datetime.fromisoformat(result["created_at"]),
                created_by=created_by,
                version_notes=version_notes,
                performance_metrics=performance_metrics,
                model_artifacts=version_data["model_artifacts"],
                is_active=True
            )
            
        except Exception as e:
            logger.error(f"Error creating model version: {e}")
            raise
    
    async def get_model_versions(self, model_id: int) -> List[ModelVersion]:
        """Get all versions for a model."""
        try:
            versions = await self.db.get_model_versions(model_id)
            return [
                ModelVersion(
                    id=v["id"],
                    model_id=v["model_id"],
                    version=v["version"],
                    created_at=datetime.fromisoformat(v["created_at"]),
                    created_by=v.get("created_by"),
                    version_notes=v.get("version_notes"),
                    performance_metrics=v.get("performance_metrics"),
                    model_artifacts=v.get("model_artifacts"),
                    is_active=v["is_active"]
                )
                for v in versions
            ]
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    async def get_active_model(self, student_id: int) -> Optional[ModelVersion]:
        """Get the active model version for a student."""
        try:
            active_model = await self.db.get_active_model(student_id)
            if not active_model:
                return None
            
            return ModelVersion(
                id=active_model["id"],
                model_id=active_model["model_id"],
                version=active_model["version"],
                created_at=datetime.fromisoformat(active_model["created_at"]),
                created_by=active_model.get("created_by"),
                version_notes=active_model.get("version_notes"),
                performance_metrics=active_model.get("performance_metrics"),
                model_artifacts=active_model.get("model_artifacts"),
                is_active=active_model["is_active"]
            )
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None
    
    async def activate_model_version(self, version_id: int, activated_by: str = None) -> bool:
        """Activate a specific model version."""
        try:
            # Get version info
            version = await self.db.get_model_version(version_id)
            if not version:
                return False
            
            # Deactivate other versions for this student
            await self.deactivate_other_versions(version["student_id"], version["model_id"])
            
            # Activate this version
            await self.db.activate_model_version(version_id)
            
            # Log the action
            await self.log_model_action(
                version["model_id"],
                "version_activated",
                new_values={"version_id": version_id, "version": version["version"]},
                performed_by=activated_by
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating model version: {e}")
            return False
    
    async def deactivate_other_versions(self, student_id: int, exclude_model_id: int = None):
        """Deactivate all other versions for a student."""
        try:
            await self.db.deactivate_other_versions(student_id, exclude_model_id)
        except Exception as e:
            logger.error(f"Error deactivating other versions: {e}")
    
    async def create_ab_test(
        self,
        student_id: int,
        model_a_id: int,
        model_b_id: int,
        test_name: str,
        description: str = None,
        traffic_split: float = 0.5,
        created_by: str = None
    ) -> ABTest:
        """Create an A/B test between two model versions."""
        try:
            ab_test_data = {
                "student_id": student_id,
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "test_name": test_name,
                "description": description,
                "traffic_split": traffic_split,
                "created_by": created_by,
                "is_active": True
            }
            
            result = await self.db.create_ab_test(ab_test_data)
            
            return ABTest(
                id=result["id"],
                student_id=student_id,
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                test_name=test_name,
                description=description,
                start_date=datetime.fromisoformat(result["start_date"]),
                end_date=None,
                is_active=True,
                traffic_split=traffic_split,
                results=None,
                created_by=created_by
            )
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise
    
    async def get_ab_test_model(self, student_id: int) -> Optional[int]:
        """Get the model ID to use for A/B testing (A or B based on traffic split)."""
        try:
            active_tests = await self.db.get_active_ab_tests(student_id)
            if not active_tests:
                return None
            
            # For simplicity, use the first active test
            # In production, you'd implement proper traffic splitting logic
            test = active_tests[0]
            
            # Simple 50/50 split based on student_id hash
            import hashlib
            hash_value = int(hashlib.md5(f"{student_id}{test['id']}".encode()).hexdigest(), 16)
            use_model_a = (hash_value % 100) < (test['traffic_split'] * 100)
            
            return test['model_a_id'] if use_model_a else test['model_b_id']
            
        except Exception as e:
            logger.error(f"Error getting A/B test model: {e}")
            return None
    
    async def record_verification_result(
        self,
        student_id: int,
        model_id: int,
        verification_result: Dict[str, Any],
        processing_time_ms: int,
        ab_test_id: int = None
    ) -> int:
        """Record a verification result for analysis."""
        try:
            result_data = {
                "student_id": student_id,
                "model_id": model_id,
                "ab_test_id": ab_test_id,
                "verification_result": verification_result,
                "processing_time_ms": processing_time_ms
            }
            
            result = await self.db.create_verification_result(result_data)
            return result["id"]
            
        except Exception as e:
            logger.error(f"Error recording verification result: {e}")
            return 0
    
    async def get_ab_test_results(self, ab_test_id: int) -> Dict[str, Any]:
        """Get A/B test results and statistics."""
        try:
            results = await self.db.get_ab_test_results(ab_test_id)
            
            if not results:
                return {"error": "No results found"}
            
            # Calculate statistics
            model_a_results = [r for r in results if r["model_id"] == results[0]["model_a_id"]]
            model_b_results = [r for r in results if r["model_id"] == results[0]["model_b_id"]]
            
            def calculate_metrics(results_list):
                if not results_list:
                    return {"count": 0, "accuracy": 0, "avg_time": 0}
                
                matches = sum(1 for r in results_list if r["verification_result"].get("match", False))
                avg_time = sum(r["processing_time_ms"] for r in results_list) / len(results_list)
                
                return {
                    "count": len(results_list),
                    "accuracy": matches / len(results_list) if results_list else 0,
                    "avg_time": avg_time,
                    "matches": matches
                }
            
            model_a_metrics = calculate_metrics(model_a_results)
            model_b_metrics = calculate_metrics(model_b_results)
            
            return {
                "ab_test_id": ab_test_id,
                "model_a_metrics": model_a_metrics,
                "model_b_metrics": model_b_metrics,
                "total_verifications": len(results),
                "test_duration_days": (datetime.now() - datetime.fromisoformat(results[0]["created_at"])).days
            }
            
        except Exception as e:
            logger.error(f"Error getting A/B test results: {e}")
            return {"error": str(e)}
    
    async def log_model_action(
        self,
        model_id: int,
        action: str,
        old_values: Dict[str, Any] = None,
        new_values: Dict[str, Any] = None,
        performed_by: str = None,
        notes: str = None
    ):
        """Log a model action to the audit trail."""
        try:
            audit_data = {
                "model_id": model_id,
                "action": action,
                "old_values": old_values,
                "new_values": new_values,
                "performed_by": performed_by,
                "notes": notes
            }
            
            await self.db.create_audit_log(audit_data)
            
        except Exception as e:
            logger.error(f"Error logging model action: {e}")
    
    async def get_model_audit_trail(self, model_id: int) -> List[Dict[str, Any]]:
        """Get audit trail for a model."""
        try:
            return await self.db.get_model_audit_trail(model_id)
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []
    
    async def get_student_model_history(self, student_id: int) -> List[Dict[str, Any]]:
        """Get complete model history for a student."""
        try:
            return await self.db.get_student_model_history(student_id)
        except Exception as e:
            logger.error(f"Error getting student model history: {e}")
            return []

# Global service instance
model_versioning_service = ModelVersioningService()
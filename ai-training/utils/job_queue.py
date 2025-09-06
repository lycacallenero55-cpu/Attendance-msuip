import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingJob:
    """Represents a training job with progress tracking."""
    
    def __init__(self, job_id: str, student_id: int, job_type: str = "training"):
        self.job_id = job_id
        self.student_id = student_id
        self.job_type = job_type
        self.status = JobStatus.PENDING
        self.progress = 0.0
        self.current_stage = "Initializing"
        self.estimated_time_remaining = None
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow()
        
        # Training metrics for real-time display
        self.training_metrics = {
            "current_epoch": 0,
            "total_epochs": 0,
            "accuracy": 0.0,
            "val_accuracy": 0.0,
            "loss": 0.0,
            "val_loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.0,
            "val_auc": 0.0,
            "learning_rate": 0.0,
            "batch_progress": "",
            "epoch_progress": ""
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "student_id": self.student_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "estimated_time_remaining": self.estimated_time_remaining,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "training_metrics": self.training_metrics
        }

class JobQueue:
    """Simple in-memory job queue for training jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.subscribers: Dict[str, asyncio.Queue] = {}
        self._running = False
        
    def create_job(self, student_id: int, job_type: str = "training") -> TrainingJob:
        """Create a new training job."""
        job_id = str(uuid.uuid4())
        job = TrainingJob(job_id, student_id, job_type)
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} for student {student_id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def update_job_progress(self, job_id: str, progress: float, stage: str, 
                          estimated_time: Optional[int] = None, training_metrics: Optional[Dict] = None):
        """Update job progress."""
        job = self.jobs.get(job_id)
        if job:
            job.progress = min(100.0, max(0.0, progress))
            job.current_stage = stage
            job.estimated_time_remaining = estimated_time
            if training_metrics:
                job.training_metrics.update(training_metrics)
            self._notify_subscribers(job)
            logger.debug(f"Job {job_id} progress: {progress:.1f}% - {stage}")
    
    def complete_job(self, job_id: str, result: Any = None):
        """Mark job as completed."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.current_stage = "Completed"
            job.end_time = datetime.utcnow()
            job.result = result
            self._notify_subscribers(job)
            logger.info(f"Job {job_id} completed successfully")
    
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.current_stage = "Failed"
            job.end_time = datetime.utcnow()
            job.error = error
            self._notify_subscribers(job)
            logger.error(f"Job {job_id} failed: {error}")
    
    def start_job(self, job_id: str):
        """Mark job as running."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.RUNNING
            job.start_time = datetime.utcnow()
            job.current_stage = "Starting"
            self._notify_subscribers(job)
            logger.info(f"Job {job_id} started")
    
    def subscribe_to_job(self, job_id: str) -> asyncio.Queue:
        """Subscribe to job updates."""
        if job_id not in self.subscribers:
            self.subscribers[job_id] = asyncio.Queue()
        return self.subscribers[job_id]
    
    def _notify_subscribers(self, job: TrainingJob):
        """Notify all subscribers of job updates."""
        if job.job_id in self.subscribers:
            try:
                self.subscribers[job.job_id].put_nowait(job.to_dict())
            except asyncio.QueueFull:
                logger.warning(f"Queue full for job {job.job_id}, dropping update")
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed/failed jobs."""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                job.created_at.timestamp() < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            if job_id in self.subscribers:
                del self.subscribers[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

# Global job queue instance
job_queue = JobQueue()
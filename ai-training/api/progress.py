from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
import json
import logging
from datetime import datetime

from utils.job_queue import job_queue

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stream/{job_id}")
async def stream_job_progress(job_id: str):
    """Stream real-time progress updates for a training job via Server-Sent Events."""
    
    # Check if job exists
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        """Generate SSE events for job progress."""
        try:
            # Subscribe to job updates
            update_queue = job_queue.subscribe_to_job(job_id)
            
            # Send initial job state
            yield f"data: {json.dumps(job.to_dict())}\n\n"
            
            # Stream updates until job is complete
            while job.status.value in ["pending", "running"]:
                try:
                    # Wait for next update with timeout
                    update = await asyncio.wait_for(update_queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    # Check if job is complete
                    if update.get("status") in ["completed", "failed", "cancelled"]:
                        break
                        
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
                except Exception as e:
                    logger.error(f"Error in SSE stream for job {job_id}: {e}")
                    break
            
            # Send final update
            final_job = job_queue.get_job(job_id)
            if final_job:
                yield f"data: {json.dumps(final_job.to_dict())}\n\n"
                
        except Exception as e:
            logger.error(f"Error in event generator for job {job_id}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of a training job."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_dict()

@router.get("/jobs/student/{student_id}")
async def get_student_jobs(student_id: int):
    """Get all jobs for a specific student."""
    student_jobs = [
        job.to_dict() for job in job_queue.jobs.values() 
        if job.student_id == student_id
    ]
    return {"jobs": student_jobs}

@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status.value in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    job.status = job.status.CANCELLED
    job.current_stage = "Cancelled"
    job.end_time = datetime.utcnow()
    job_queue._notify_subscribers(job)
    
    return {"message": "Job cancelled successfully"}
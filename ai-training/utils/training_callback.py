"""
Custom training callback for real-time metrics streaming
"""
import logging
from tensorflow.keras.callbacks import Callback
from utils.job_queue import job_queue

logger = logging.getLogger(__name__)

class RealTimeMetricsCallback(Callback):
    """Custom callback to stream training metrics in real-time"""
    
    def __init__(self, job_id: str, total_epochs: int):
        super().__init__()
        self.job_id = job_id
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        logger.info(f"Training started for job {self.job_id}")
        job_queue.update_job_progress(
            self.job_id, 
            20.0, 
            "Training started", 
            training_metrics={
                "total_epochs": self.total_epochs,
                "current_epoch": 0,
                "epoch_progress": "0/0"
            }
        )
        logger.info(f"RealTimeMetricsCallback initialized for job {self.job_id} with {self.total_epochs} epochs")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.current_epoch = epoch + 1
        self.current_batch = 0
        
        # Calculate progress based on epochs
        epoch_progress = (self.current_epoch / self.total_epochs) * 60.0 + 20.0  # 20-80% range
        
        logger.info(f"Epoch {self.current_epoch} starting, progress: {epoch_progress}%")
        
        job_queue.update_job_progress(
            self.job_id,
            epoch_progress,
            f"Epoch {self.current_epoch}/{self.total_epochs}",
            training_metrics={
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "epoch_progress": f"{self.current_epoch}/{self.total_epochs}"
            }
        )
        logger.info(f"Epoch {self.current_epoch} progress update sent to job queue")
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch"""
        self.current_batch = batch + 1
        
        # Extract metrics from logs
        metrics = {}
        if logs:
            metrics.update({
                "accuracy": float(logs.get('accuracy', 0.0)),
                "loss": float(logs.get('loss', 0.0)),
                "precision": float(logs.get('precision', 0.0)),
                "recall": float(logs.get('recall', 0.0)),
                "auc": float(logs.get('auc', 0.0)),
                "learning_rate": float(logs.get('learning_rate', 0.0)),
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_batch": self.current_batch
            })
            
            # Add batch progress
            if 'size' in logs:
                batch_size = logs['size']
                metrics["batch_progress"] = f"Batch {self.current_batch} (size: {batch_size})"
        
        # Calculate progress within epoch
        if self.total_batches > 0:
            batch_progress = (self.current_batch / self.total_batches) * (60.0 / self.total_epochs)
            epoch_progress = (self.current_epoch / self.total_epochs) * 60.0 + 20.0
            total_progress = epoch_progress + batch_progress
        else:
            total_progress = (self.current_epoch / self.total_epochs) * 60.0 + 20.0
        
        logger.info(f"Batch {self.current_batch} completed, progress: {total_progress}%, metrics: {metrics}")
        
        job_queue.update_job_progress(
            self.job_id,
            total_progress,
            f"Epoch {self.current_epoch}/{self.total_epochs} - Batch {self.current_batch}",
            training_metrics=metrics
        )
        logger.info(f"Batch {self.current_batch} progress update sent to job queue")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        # Extract validation metrics
        metrics = {
            "current_epoch": self.current_epoch,
            "epoch_progress": f"{self.current_epoch}/{self.total_epochs}"
        }
        
        if logs:
            metrics.update({
                "accuracy": float(logs.get('accuracy', 0.0)),
                "val_accuracy": float(logs.get('val_accuracy', 0.0)),
                "loss": float(logs.get('loss', 0.0)),
                "val_loss": float(logs.get('val_loss', 0.0)),
                "precision": float(logs.get('precision', 0.0)),
                "recall": float(logs.get('recall', 0.0)),
                "auc": float(logs.get('auc', 0.0)),
                "val_auc": float(logs.get('val_auc', 0.0)),
                "learning_rate": float(logs.get('learning_rate', 0.0))
            })
        
        # Calculate progress
        epoch_progress = ((epoch + 1) / self.total_epochs) * 60.0 + 20.0
        
        job_queue.update_job_progress(
            self.job_id,
            epoch_progress,
            f"Epoch {self.current_epoch}/{self.total_epochs} completed",
            training_metrics=metrics
        )
        
        logger.info(f"Epoch {self.current_epoch}/{self.total_epochs} completed for job {self.job_id}")
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        logger.info(f"Training completed for job {self.job_id}")
        job_queue.update_job_progress(
            self.job_id,
            80.0,
            "Training completed, saving model...",
            training_metrics={
                "current_epoch": self.current_epoch,
                "epoch_progress": f"{self.current_epoch}/{self.total_epochs}",
                "accuracy": float(logs.get('accuracy', 0.0)) if logs else 0.0,
                "val_accuracy": float(logs.get('val_accuracy', 0.0)) if logs else 0.0,
                "loss": float(logs.get('loss', 0.0)) if logs else 0.0,
                "val_loss": float(logs.get('val_loss', 0.0)) if logs else 0.0
            }
        )
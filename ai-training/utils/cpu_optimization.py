import tensorflow as tf
import os
import logging
import numpy as np  

logger = logging.getLogger(__name__)


def configure_tensorflow_for_cpu():
    """Configure TensorFlow for optimal CPU performance on Ryzen processors"""
    
    # Set CPU as the only visible device
    try:
        # Hide GPU devices
        tf.config.set_visible_devices([], 'GPU')
        
        # Get CPU devices
        cpus = tf.config.list_physical_devices('CPU')
        if cpus:
            logger.info(f"Found {len(cpus)} CPU devices")
                
    except RuntimeError as e:
        logger.warning(f"Could not configure CPU devices: {e}")
    
    # Set number of inter-op and intra-op threads for Ryzen
    # Ryzen 5 3400G has 4 cores, 8 threads
    num_cores = os.cpu_count() or 4
    optimal_threads = min(num_cores - 1, 6)  # Leave one core free for system
    
    tf.config.threading.set_inter_op_parallelism_threads(optimal_threads)
    tf.config.threading.set_intra_op_parallelism_threads(optimal_threads)
    
    # Enable XLA JIT compilation for faster CPU execution
    tf.config.optimizer.set_jit(True)
    
    # Set memory allocation options
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
    
    # AMD specific optimizations
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
    
    logger.info(f"TensorFlow configured for CPU with {optimal_threads} threads")
    return optimal_threads


def get_optimal_batch_size(model_size='medium', ram_gb=8):
    """Get optimal batch size based on model size and available RAM"""
    
    # Base recommendations for Ryzen 5 3400G
    batch_sizes = {
        'small': {'8gb': 32, '16gb': 64, '32gb': 128},
        'medium': {'8gb': 16, '16gb': 32, '32gb': 64},
        'large': {'8gb': 8, '16gb': 16, '32gb': 32}
    }
    
    # Determine RAM category
    if ram_gb <= 8:
        ram_key = '8gb'
    elif ram_gb <= 16:
        ram_key = '16gb'
    else:
        ram_key = '32gb'
    
    return batch_sizes.get(model_size, batch_sizes['medium'])[ram_key]


class CPUDataGenerator(tf.keras.utils.Sequence):
    """Optimized data generator for CPU training"""
    
    def __init__(self, x_a, x_b, y, batch_size=16, shuffle=True):
        self.x_a = x_a
        self.x_b = x_b
        self.y = y
        self.batch_size = int(batch_size)  # Ensure batch_size is integer
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.y))
        batch_indices = self.indices[start:end]
        
        batch_x_a = self.x_a[batch_indices]
        batch_x_b = self.x_b[batch_indices]
        batch_y = self.y[batch_indices]
        
        return [batch_x_a, batch_x_b], batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_cpu_optimized_model_checkpoint():
    """Create model checkpoint callback optimized for CPU"""
    return tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model_{epoch:02d}_{val_auc:.3f}.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        save_freq='epoch',
        options=tf.saved_model.SaveOptions(
            experimental_io_device='/job:localhost'  # Save on CPU
        ) if hasattr(tf.saved_model, 'SaveOptions') else None
    )


def profile_model_performance(model, test_data):
    """Profile model performance on CPU"""
    import time
    
    # Warmup
    _ = model.predict(test_data[:2], verbose=0)
    
    # Measure inference time
    num_samples = len(test_data)
    start_time = time.time()
    predictions = model.predict(test_data, verbose=0, batch_size=1)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples * 1000  # ms per sample
    
    return {
        'total_inference_time': total_time,
        'avg_inference_time_ms': avg_time,
        'samples_per_second': num_samples / total_time
    }
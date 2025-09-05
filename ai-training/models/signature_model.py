import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import settings
import logging
from PIL import Image


logger = logging.getLogger(__name__)

class SignatureVerificationModel:
    def __init__(self):
        self.model = None
        self.embedding_model = None
        self.image_size = settings.MODEL_IMAGE_SIZE
        self.batch_size = settings.MODEL_BATCH_SIZE
        self.epochs = settings.MODEL_EPOCHS
        self.learning_rate = settings.MODEL_LEARNING_RATE
    
    def create_siamese_network(self):
        """Create a Siamese neural network for signature verification"""
        
        # Input layers
        input_a = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_a')
        input_b = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_b')
        
        # Shared CNN backbone - optimized for CPU
        def create_embedding_branch():
            model = keras.Sequential([
                # First block - reduced filters for CPU efficiency
                layers.Conv2D(24, (5, 5), activation='relu', padding='same',
                            input_shape=(self.image_size, self.image_size, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second block
                layers.Conv2D(48, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third block
                layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                
                # Fourth block - signature-specific features
                layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                
                # Dense layers for feature extraction
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation=None)  # No activation for embeddings
            ], name='embedding_branch')
            return model
        
        # Create shared embedding network
        embedding_network = create_embedding_branch()
        
        # Get embeddings for both inputs
        embedding_a = embedding_network(input_a)
        embedding_b = embedding_network(input_b)
        
        # Compute L2 distance between embeddings (no Lambda layer)
        # Using Subtract + custom layer for compatibility
        diff = layers.Subtract(name='embedding_diff')([embedding_a, embedding_b])
        
        # Custom layer to compute absolute difference without Lambda
        class AbsDiffLayer(layers.Layer):
            def call(self, inputs):
                return tf.abs(inputs)
        
        abs_diff = AbsDiffLayer(name='abs_diff')(diff)
        
        # Classification head with residual connections
        x = layers.Dense(32, activation='relu')(abs_diff)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Add skip connection for better gradient flow
        x2 = layers.Dense(16, activation='relu')(x)
        x2 = layers.BatchNormalization()(x2)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='similarity_score')(x2)
        
        # Create the model
        model = keras.Model(inputs=[input_a, input_b], outputs=output, name='signature_verification')
        
        # Use AdamW optimizer for better generalization
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.0001
        )
        
        # Compile with additional metrics
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        # Create standalone embedding model for inference
        single_input = layers.Input(shape=(self.image_size, self.image_size, 3), name='single_input')
        single_embedding = embedding_network(single_input)
        self.embedding_model = keras.Model(inputs=single_input, outputs=single_embedding, name='embedding_model')

        return model
    
    def prepare_augmented_data(self, all_images, all_labels):
        """Prepare balanced training data from augmented images"""
        # FIX: Ensure all images are properly converted to numpy arrays with consistent shape
        processed_images = []
        for img in all_images:
            if isinstance(img, Image.Image):
                # Convert PIL to numpy and ensure correct shape
                img_array = np.array(img)
                if img_array.ndim == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
            
                # Ensure image is the correct size
                if img_array.shape[:2] != (self.image_size, self.image_size):
                    # Resize using PIL for consistency
                    pil_img = Image.fromarray(img_array.astype(np.uint8))
                    pil_img = pil_img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                    img_array = np.array(pil_img)
            
                # Normalize to [0, 1]
                img_array = img_array.astype(np.float32) / 255.0
                processed_images.append(img_array)
            else:
                # Already numpy array
                img_array = np.array(img)
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
            
                # Ensure correct shape
                if img_array.shape[:2] != (self.image_size, self.image_size):
                    # Use cv2 for resizing numpy arrays
                    import cv2
                    img_array = cv2.resize(img_array, (self.image_size, self.image_size))
                    if img_array.ndim == 2:  # cv2 might return grayscale
                        img_array = np.stack([img_array] * 3, axis=-1)
            
                # Normalize if needed
                if img_array.max() > 1.0:
                    img_array = img_array.astype(np.float32) / 255.0
                processed_images.append(img_array)
    
        # Now create the numpy array with consistent shapes
        images = np.array(processed_images, dtype=np.float32)
        labels = np.array(all_labels)
    
        logger.info(f"Processed images shape: {images.shape}, Labels shape: {labels.shape}")
    
        pairs = []
        pair_labels = []
    
        # Get indices
        genuine_indices = np.where(labels == True)[0]
        forged_indices = np.where(labels == False)[0]
    
        # Balance the dataset
        num_genuine = len(genuine_indices)
        num_forged = len(forged_indices)
    
        # Create positive pairs (same class)
        for i in range(num_genuine):
            for j in range(i + 1, min(i + 3, num_genuine)):
                idx1, idx2 = genuine_indices[i], genuine_indices[j]
                pairs.append([images[idx1], images[idx2]])
                pair_labels.append(1)
        
        # Create negative pairs
        max_neg_pairs = len(pairs) * 2
        neg_pair_count = 0
        
        for i in range(num_genuine):
            for j in range(min(3, num_forged)):
                if neg_pair_count >= max_neg_pairs:
                    break
                genuine_idx = genuine_indices[i]
                forged_idx = forged_indices[j % num_forged]
                pairs.append([images[genuine_idx], images[forged_idx]])
                pair_labels.append(0)
                neg_pair_count += 1
    
        # Convert to arrays
        pairs = np.array(pairs, dtype=np.float32)
        pair_labels = np.array(pair_labels, dtype=np.float32)
    
        # Shuffle
        indices = np.arange(len(pairs))
        np.random.RandomState(42).shuffle(indices)
        pairs = pairs[indices]
        pair_labels = pair_labels[indices]
    
        # Split into input arrays
        input_a = pairs[:, 0]
        input_b = pairs[:, 1]
    
        logger.info(f"Created {len(pairs)} training pairs: {np.sum(pair_labels)} positive, {len(pairs) - np.sum(pair_labels)} negative")
        logger.info(f"Input shapes: input_a={input_a.shape}, input_b={input_b.shape}, labels={pair_labels.shape}")
    
        return input_a, input_b, pair_labels
    
    def train_with_augmented_data(self, all_images, all_labels, validation_split=0.2):
        """Train the signature verification model with augmented data"""
        try:
            # Prepare balanced data
            input_a, input_b, labels = self.prepare_augmented_data(all_images, all_labels)
            
            # Create model
            self.model = self.create_siamese_network()
            
            # Callbacks for CPU-optimized training
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='temp_best_model.keras',
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    verbose=0
                )
            ]
            
            # Custom training configuration for CPU
            # Reduce batch size for CPU memory efficiency
            cpu_batch_size = min(self.batch_size, 16)
            
            # Train the model
            history = self.model.fit(
                [input_a, input_b],
                labels,
                batch_size=cpu_batch_size,
                epochs=self.epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
            )
            
            # Load best model if checkpoint exists
            try:
                self.model = keras.models.load_model('temp_best_model.keras', safe_mode=False)
                import os
                if os.path.exists('temp_best_model.keras'):
                    os.remove('temp_best_model.keras')
            except:
                pass
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def compute_centroid_and_adaptive_threshold(self, genuine_images, forged_images=None):
        """Compute centroid and adaptive threshold using statistical methods"""
        if self.embedding_model is None:
            raise ValueError("Embedding model not available")
        
        # Get genuine embeddings
        genuine_embeddings = self.embed_images(genuine_images)
        
        # Compute centroid (prototype)
        centroid = np.mean(genuine_embeddings, axis=0)
        
        # Calculate distances from genuine samples to centroid
        genuine_distances = np.linalg.norm(genuine_embeddings - centroid, axis=1)
        
        # Statistical threshold calculation
        mean_dist = np.mean(genuine_distances)
        std_dist = np.std(genuine_distances)
        
        # Adaptive threshold strategies
        if forged_images is not None and len(forged_images) > 0:
            # If we have forged samples, use them for better calibration
            forged_embeddings = self.embed_images(forged_images)
            forged_distances = np.linalg.norm(forged_embeddings - centroid, axis=1)
            
            # Find optimal threshold using ROC curve analysis
            threshold = self._find_optimal_threshold(genuine_distances, forged_distances)
        else:
            # Conservative threshold: mean + 2*std covers ~95% of genuine signatures
            threshold = mean_dist + 2.0 * std_dist
            
            # Apply bounds to prevent extreme thresholds
            max_threshold = mean_dist + 3.0 * std_dist
            min_threshold = mean_dist + 0.5 * std_dist
            threshold = np.clip(threshold, min_threshold, max_threshold)
        
        logger.info(f"Threshold calibration: mean={mean_dist:.3f}, std={std_dist:.3f}, threshold={threshold:.3f}")
        
        return centroid.tolist(), float(threshold)
    
    def _find_optimal_threshold(self, genuine_distances, forged_distances):
        """Find optimal threshold using Equal Error Rate (EER)"""
        all_distances = np.concatenate([genuine_distances, forged_distances])
        thresholds = np.percentile(all_distances, np.linspace(5, 95, 50))
        
        best_threshold = thresholds[0]
        best_score = float('inf')
        
        for threshold in thresholds:
            # False Rejection Rate (genuine rejected)
            frr = np.sum(genuine_distances > threshold) / len(genuine_distances)
            # False Acceptance Rate (forged accepted)
            far = np.sum(forged_distances <= threshold) / len(forged_distances)
            
            # Find threshold where FRR ≈ FAR (Equal Error Rate)
            eer_score = abs(frr - far)
            
            # Also consider overall error
            total_error = 0.5 * (frr + far)
            combined_score = eer_score + 0.1 * total_error
            
            if combined_score < best_score:
                best_score = combined_score
                best_threshold = threshold
        
        return float(best_threshold)
    
    def embed_images(self, images):
        """Generate embeddings for a list of PIL images"""
        if self.embedding_model is None:
            raise ValueError("Embedding model not available")
        
        batch = []
        for img in images:
            # Handle both PIL and numpy arrays
            if hasattr(img, 'convert'):  # PIL Image
                arr = np.array(img.convert('RGB'))
            else:
                arr = np.array(img)
            
            # Ensure correct shape
            if arr.shape[-1] != 3:
                if len(arr.shape) == 2:  # Grayscale
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.shape[-1] == 4:  # RGBA
                    arr = arr[:, :, :3]
            
            arr = tf.convert_to_tensor(arr, dtype=tf.float32)
            arr = self.preprocess_image(arr)
            batch.append(arr)
        
        batch_tensor = tf.stack(batch, axis=0)
        embeddings = self.embedding_model.predict(batch_tensor, verbose=0)
        return embeddings
    
    def preprocess_image(self, image):
        """Enhanced preprocessing for better feature extraction"""
        # Resize image
        image = tf.image.resize(image, [self.image_size, self.image_size])
        
        # Normalize to [-1, 1] for better gradient flow
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        
        return image
    
    def save_embedding_model(self, filepath):
        """Save the embedding model separately"""
        if self.embedding_model is None:
            raise ValueError("No embedding model to save")
        self.embedding_model.save(filepath)
        logger.info(f"Embedding model saved to {filepath}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath, safe_mode=False)
        
        # Reconstruct embedding model
        try:
            embedding_branch = self.model.get_layer('embedding_branch')
            single_input = layers.Input(shape=(self.image_size, self.image_size, 3))
            single_embedding = embedding_branch(single_input)
            self.embedding_model = keras.Model(inputs=single_input, outputs=single_embedding)
        except Exception as e:
            logger.error(f"Could not reconstruct embedding model: {e}")
            self.embedding_model = None
        
        logger.info(f"Model loaded from {filepath}")

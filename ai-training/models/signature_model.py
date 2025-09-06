# signature_model.py - FIXED VERSION (No Lambda Layers)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import settings
import logging
from typing import List, Tuple, Optional, Union
from PIL import Image
import os

logger = logging.getLogger(__name__)

class SignatureVerificationModel:
    """Enhanced Siamese Neural Network for Signature Verification with Prototype Learning"""
    
    def __init__(self):
        self.model = None
        self.embedding_model = None
        self.image_size = settings.MODEL_IMAGE_SIZE
        self.batch_size = settings.MODEL_BATCH_SIZE
        self.epochs = settings.MODEL_EPOCHS
        self.learning_rate = settings.MODEL_LEARNING_RATE
        
        # Enhanced training parameters
        self.use_mixed_precision = False  # Disabled for CPU
        self.use_gradient_accumulation = True
        self.gradient_accumulation_steps = 4
        
    def create_siamese_network(self):
        """Create an improved Siamese neural network for signature verification"""
        
        # Input layers
        input_a = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_a')
        input_b = layers.Input(shape=(self.image_size, self.image_size, 3), name='input_b')
        
        # Enhanced embedding branch with attention mechanism
        def create_embedding_branch():
            base = keras.applications.MobileNetV2(
                input_shape=(self.image_size, self.image_size, 3),
                include_top=False,
                weights='imagenet')
            
            # Fine-tune more layers for signature-specific features
            for layer in base.layers[:-40]:  # Unfreeze last 40 layers
                layer.trainable = False
            
            x = layers.Input(shape=(self.image_size, self.image_size, 3))
            y = keras.applications.mobilenet_v2.preprocess_input(x)
            y = base(y, training=False)
            
            # Add spatial attention mechanism
            attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(y)
            y = layers.Multiply()([y, attention])
            
            y = layers.GlobalAveragePooling2D()(y)
            
            # Deeper feature extraction with residual connections
            y1 = layers.Dense(512, activation='relu')(y)
            y1 = layers.BatchNormalization()(y1)
            y1 = layers.Dropout(0.4)(y1)
            
            y2 = layers.Dense(256, activation='relu')(y1)
            y2 = layers.BatchNormalization()(y2)
            y2 = layers.Dropout(0.3)(y2)
            
            # Residual connection
            y1_proj = layers.Dense(256, activation='linear')(y1)
            y2 = layers.Add()([y2, y1_proj])
            
            y3 = layers.Dense(128, activation='relu')(y2)
            y3 = layers.BatchNormalization()(y3)
            
            # Use LayerNormalization instead of Lambda for L2 normalization
            y3 = layers.LayerNormalization(axis=1)(y3)
            
            model = keras.Model(inputs=x, outputs=y3, name='embedding_branch')
            return model
        
        # Create shared embedding network
        embedding_network = create_embedding_branch()
        
        # Get embeddings for both inputs
        embedding_a = embedding_network(input_a)
        embedding_b = embedding_network(input_b)
        
        # Use simple concatenation instead of complex distance metrics
        # This avoids Lambda layers that cause serialization issues
        merged = layers.Concatenate()([embedding_a, embedding_b])
        
        # Enhanced classification head
        output = layers.Dense(256, activation='relu')(merged)
        output = layers.BatchNormalization()(output)
        output = layers.Dropout(0.4)(output)
        output = layers.Dense(128, activation='relu')(output)
        output = layers.BatchNormalization()(output)
        output = layers.Dropout(0.3)(output)
        output = layers.Dense(64, activation='relu')(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(1, activation='sigmoid', name='similarity_score')(output)
        
        # Create the model
        model = keras.Model(inputs=[input_a, input_b], outputs=output, name='signature_verification')
        
        # Use simple Adam optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Custom metrics including AUC-ROC and AUC-PR
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc', curve='ROC'),
                keras.metrics.AUC(name='auc_pr', curve='PR')
            ]
        )
        
        # Store embedding model separately
        self.embedding_model = embedding_network
        
        return model
    
    def compute_centroid_and_adaptive_threshold(self, genuine_images: List, forged_images: Optional[List] = None, 
                                               val_genuine_images: Optional[List] = None, 
                                               val_forged_images: Optional[List] = None) -> Tuple[List[float], float]:
        """Compute centroid and adaptive threshold using EER optimization
        
        Args:
            genuine_images: List of genuine signature images for centroid computation
            forged_images: Optional list of forged signature images for threshold computation
            val_genuine_images: Optional validation genuine images (preferred for threshold)
            val_forged_images: Optional validation forged images (preferred for threshold)
            
        Returns:
            Tuple of (centroid as list, threshold as float)
        """
        if not genuine_images:
            raise ValueError("No genuine images provided")
        
        # Get embeddings for genuine samples to compute centroid
        embeddings = self.embed_images(genuine_images)
        centroid = np.mean(embeddings, axis=0)
        
        # Use validation data for threshold computation if available (prevents overfitting)
        if val_genuine_images is not None and val_forged_images is not None:
            logger.info("Using validation data for threshold computation")
            val_genuine_embeddings = self.embed_images(val_genuine_images)
            val_forged_embeddings = self.embed_images(val_forged_images)
            
            val_genuine_dists = np.linalg.norm(val_genuine_embeddings - centroid, axis=1)
            val_forged_dists = np.linalg.norm(val_forged_embeddings - centroid, axis=1)
            
            threshold = self._find_eer_threshold(val_genuine_dists, val_forged_dists)
            logger.info(f"Computed validation-based threshold: {threshold:.4f}")
            
        elif forged_images and len(forged_images) > 0:
            # Fallback to training data if no validation data
            logger.warning("Using training data for threshold computation - may cause overfitting")
            genuine_dists = np.linalg.norm(embeddings - centroid, axis=1)
            forged_embeddings = self.embed_images(forged_images)
            forged_dists = np.linalg.norm(forged_embeddings - centroid, axis=1)
            
            threshold = self._find_eer_threshold(genuine_dists, forged_dists)
            logger.info(f"Computed training-based threshold: {threshold:.4f}")
        else:
            # Use statistical approach if no forged samples
            genuine_dists = np.linalg.norm(embeddings - centroid, axis=1)
            threshold = float(np.percentile(genuine_dists, 95) * 1.2)
            logger.info(f"Computed statistical threshold: {threshold:.4f}")
        
        return centroid.tolist(), threshold
    
    def _find_eer_threshold(self, genuine_dists: np.ndarray, forged_dists: np.ndarray) -> float:
        """Find Equal Error Rate threshold"""
        all_dists = np.concatenate([genuine_dists, forged_dists])
        
        best_threshold = 0.5
        best_eer = 1.0
        
        for percentile in np.linspace(5, 95, 50):
            threshold = np.percentile(all_dists, percentile)
            
            # Calculate FAR and FRR
            frr = np.mean(genuine_dists > threshold)  # False Rejection Rate
            far = np.mean(forged_dists <= threshold)  # False Acceptance Rate
            
            # EER is when FAR ≈ FRR
            eer = abs(frr - far)
            
            if eer < best_eer:
                best_eer = eer
                best_threshold = threshold
        
        # Add safety margin (10% buffer)
        return float(best_threshold * 1.1)
    
    def evaluate_model_comprehensive(self, genuine_images: List, forged_images: List, 
                                   threshold: float = None) -> dict:
        """Comprehensive model evaluation with multiple metrics"""
        from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Get embeddings
        genuine_embeddings = self.embed_images(genuine_images)
        forged_embeddings = self.embed_images(forged_images)
        
        # Compute centroid
        centroid = np.mean(genuine_embeddings, axis=0)
        
        # Compute distances
        genuine_dists = np.linalg.norm(genuine_embeddings - centroid, axis=1)
        forged_dists = np.linalg.norm(forged_embeddings - centroid, axis=1)
        
        # Manual ROC computation without sklearn
        all_dists = np.concatenate([genuine_dists, forged_dists])
        all_labels = np.concatenate([np.ones(len(genuine_dists)), np.zeros(len(forged_dists))])
        
        # Sort by distance (ascending)
        sorted_indices = np.argsort(all_dists)
        sorted_dists = all_dists[sorted_indices]
        sorted_labels = all_labels[sorted_indices]
        
        # Compute ROC curve manually
        thresholds = np.unique(sorted_dists)
        tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))
        
        for i, thresh in enumerate(thresholds):
            # True Positive Rate (sensitivity)
            tp = np.sum((sorted_labels == 1) & (sorted_dists <= thresh))
            fn = np.sum((sorted_labels == 1) & (sorted_dists > thresh))
            tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Positive Rate (1 - specificity)
            fp = np.sum((sorted_labels == 0) & (sorted_dists <= thresh))
            tn = np.sum((sorted_labels == 0) & (sorted_dists > thresh))
            fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Compute AUC manually (trapezoidal rule)
        roc_auc = 0.0
        for i in range(1, len(fpr)):
            roc_auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        
        # Find EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fnr - fpr))
        eer = fpr[eer_idx]
        
        # Use provided threshold or compute optimal
        if threshold is None:
            threshold = thresholds[eer_idx]
        
        # Compute predictions at threshold
        genuine_pred = (genuine_dists <= threshold).astype(int)
        forged_pred = (forged_dists <= threshold).astype(int)
        y_pred = np.concatenate([genuine_pred, forged_pred])
        y_true = np.concatenate([np.ones(len(genuine_dists)), np.zeros(len(forged_dists))])
        
        # Compute classification metrics manually
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision_at_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_at_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_at_thresh = 2 * (precision_at_thresh * recall_at_thresh) / (precision_at_thresh + recall_at_thresh) if (precision_at_thresh + recall_at_thresh) > 0 else 0
        
        # Compute PR-AUC manually
        precision_vals = np.zeros(len(thresholds))
        recall_vals = np.zeros(len(thresholds))
        
        for i, thresh in enumerate(thresholds):
            tp = np.sum((sorted_labels == 1) & (sorted_dists <= thresh))
            fp = np.sum((sorted_labels == 0) & (sorted_dists <= thresh))
            fn = np.sum((sorted_labels == 1) & (sorted_dists > thresh))
            
            precision_vals[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_vals[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Compute PR-AUC using trapezoidal rule
        pr_auc = 0.0
        for i in range(1, len(recall_vals)):
            pr_auc += (recall_vals[i] - recall_vals[i-1]) * (precision_vals[i] + precision_vals[i-1]) / 2
        
        # Compute FAR and FRR at threshold
        far = np.mean(forged_dists <= threshold)  # False Acceptance Rate
        frr = np.mean(genuine_dists > threshold)  # False Rejection Rate
        
        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'eer': float(eer),
            'accuracy': float(accuracy),
            'precision': float(precision_at_thresh),
            'recall': float(recall_at_thresh),
            'f1_score': float(f1_at_thresh),
            'far': float(far),
            'frr': float(frr),
            'threshold': float(threshold),
            'genuine_distances': {
                'mean': float(np.mean(genuine_dists)),
                'std': float(np.std(genuine_dists)),
                'min': float(np.min(genuine_dists)),
                'max': float(np.max(genuine_dists))
            },
            'forged_distances': {
                'mean': float(np.mean(forged_dists)),
                'std': float(np.std(forged_dists)),
                'min': float(np.min(forged_dists)),
                'max': float(np.max(forged_dists))
            }
        }
    
    def embed_images(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """Generate normalized embeddings for images"""
        if self.embedding_model is None:
            raise ValueError("Embedding model not available")
        
        batch = []
        for img in images:
            # Handle both PIL and numpy arrays
            if hasattr(img, 'convert'):  # PIL Image
                arr = np.array(img.convert('RGB'))
            else:
                arr = img
            
            # Ensure correct shape and dtype
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            
            # FIXED: Ensure float32 dtype and proper range [0, 255]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            
            # Ensure values are in [0, 255] range
            if arr.max() <= 1.0:
                arr = arr * 255.0
            
            # Preprocess
            arr = tf.convert_to_tensor(arr, dtype=tf.float32)
            arr = self.preprocess_image(arr)
            batch.append(arr)
        
        batch_tensor = tf.stack(batch, axis=0)
        embeddings = self.embedding_model.predict(batch_tensor, verbose=0)
        
        # Apply L2 normalization manually since we removed Lambda layer
        embeddings = tf.nn.l2_normalize(embeddings, axis=1).numpy()
        
        return embeddings
    
    def prepare_augmented_data(self, all_images, all_labels):
        """Prepare training data from augmented images and labels"""
        # Convert PIL images to numpy arrays with consistent dtype and 3 channels
        img_arrays = []
        for img in all_images:
            # Handle both PIL and numpy arrays
            if hasattr(img, 'convert'):  # PIL Image
                arr = np.array(img.convert('RGB'))
            else:
                arr = img
            
            # Ensure correct shape
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            
            # FIXED: Ensure float32 dtype and proper range [0, 255]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            
            # Ensure values are in [0, 255] range
            if arr.max() <= 1.0:
                arr = arr * 255.0
            
            img_arrays.append(arr)
    
        images = np.stack(img_arrays, axis=0)
        labels = np.array(all_labels, dtype=bool)
    
        # Debug: print shapes and dtypes
        logger.info(f"prepare_augmented_data: images shape={images.shape}, dtype={images.dtype}")
        logger.info(f"prepare_augmented_data: labels shape={labels.shape}, dtype={labels.dtype}")
    
        # Create pairs for Siamese training
        pairs = []
        pair_labels = []
    
        # Generate positive pairs (same class)
        genuine_indices = np.where(labels == True)[0]
        forged_indices = np.where(labels == False)[0]
    
        # Genuine pairs
        for i in range(len(genuine_indices)):
            for j in range(i + 1, min(i + 3, len(genuine_indices))):  # Limit pairs per image
                idx1, idx2 = genuine_indices[i], genuine_indices[j]
                pairs.append([images[idx1], images[idx2]])
                pair_labels.append(1)  # Similar
    
        # Forged pairs - FIXED: These should be labeled as 0 (different), not 1 (similar)
        for i in range(len(forged_indices)):
            for j in range(i + 1, min(i + 2, len(forged_indices))):  # Limit pairs per image
                idx1, idx2 = forged_indices[i], forged_indices[j]
                pairs.append([images[idx1], images[idx2]])
                pair_labels.append(0)  # FIXED: Forged-forged pairs are different signatures, not similar
    
        # Generate negative pairs (different classes)
        for i in range(min(len(genuine_indices), len(forged_indices))):
            for j in range(min(2, len(forged_indices))):  # Limit negative pairs
                genuine_idx = genuine_indices[i]
                forged_idx = forged_indices[j]
                pairs.append([images[genuine_idx], images[forged_idx]])
                pair_labels.append(0)  # Different
    
        # Convert to numpy arrays with consistent dtype
        pairs = np.stack(pairs, axis=0).astype(np.float32)
        pair_labels = np.array(pair_labels, dtype=np.float32)
    
        # Debug: print final shapes and dtypes
        logger.info(f"prepare_augmented_data final: pairs shape={pairs.shape}, dtype={pairs.dtype}")
        logger.info(f"prepare_augmented_data final: pair_labels shape={pair_labels.shape}, dtype={pair_labels.dtype}")
    
        # Split into input_a and input_b
        input_a = pairs[:, 0]
        input_b = pairs[:, 1]
    
        # Debug: print split shapes
        logger.info(f"prepare_augmented_data split: input_a shape={input_a.shape}, dtype={input_a.dtype}")
        logger.info(f"prepare_augmented_data split: input_b shape={input_b.shape}, dtype={input_b.dtype}")
    
        return input_a, input_b, pair_labels
    
    def prepare_augmented_pairs(self, images: List, labels: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training pairs from augmented data with balanced sampling"""
        genuine_imgs = [img for img, lbl in zip(images, labels) if lbl]
        forged_imgs = [img for img, lbl in zip(images, labels) if not lbl]
    
        pairs_a = []
        pairs_b = []
        pair_labels = []
    
        # Generate positive pairs (genuine-genuine)
        num_positive = min(len(genuine_imgs) * 10, 5000)  # Cap at 5000 pairs
        for _ in range(num_positive):
            idx1, idx2 = np.random.choice(len(genuine_imgs), 2, replace=True)
            pairs_a.append(genuine_imgs[idx1])
            pairs_b.append(genuine_imgs[idx2])
            pair_labels.append(1)
    
        # Generate negative pairs (genuine-forged)
        if forged_imgs:
            num_negative = num_positive
            for _ in range(num_negative):
                idx1 = np.random.choice(len(genuine_imgs))
                idx2 = np.random.choice(len(forged_imgs))
                pairs_a.append(genuine_imgs[idx1])
                pairs_b.append(forged_imgs[idx2])
                pair_labels.append(0)
    
        # Generate hard negative pairs (forged-forged) - FIXED: These are different signatures
        if forged_imgs and len(forged_imgs) > 1:
            num_hard_negative = num_positive // 10  # 10% hard negatives
            for _ in range(num_hard_negative):
                idx1, idx2 = np.random.choice(len(forged_imgs), 2, replace=False)
                pairs_a.append(forged_imgs[idx1])
                pairs_b.append(forged_imgs[idx2])
                pair_labels.append(0)  # FIXED: Forged-forged pairs are different signatures
    
        # Convert to numpy arrays with consistent dtype
        pairs_a = self._process_batch(pairs_a)
        pairs_b = self._process_batch(pairs_b)
        pair_labels = np.array(pair_labels, dtype=np.float32)
    
        # Debug: print shapes and dtypes
        logger.info(f"prepare_augmented_pairs shapes: pairs_a={pairs_a.shape}, pairs_b={pairs_b.shape}, labels={pair_labels.shape}")
        logger.info(f"prepare_augmented_pairs dtypes: pairs_a={pairs_a.dtype}, pairs_b={pairs_b.dtype}, labels={pair_labels.dtype}")
    
        # Shuffle
        indices = np.arange(len(pair_labels))
        np.random.shuffle(indices)
    
        return pairs_a[indices], pairs_b[indices], pair_labels[indices]
    
    def preprocess_image(self, image):
        """Preprocess image for the model"""
        # Ensure input is float32 tensor
        if isinstance(image, np.ndarray):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        else:
            image = tf.cast(image, tf.float32)
        
        # Resize image
        image = tf.image.resize(image, [self.image_size, self.image_size])
        # MobileNetV2 preprocessing expects float [-1,1]
        image = (image / 127.5) - 1.0
        return image
    
    def _process_batch(self, images: List) -> np.ndarray:
        """Process batch of images to tensor"""
        processed = []
        for img in images:
            if hasattr(img, 'convert'):  # PIL Image
                arr = np.array(img.convert('RGB'))
            else:
                arr = img
            
            # Ensure correct shape
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            
            # FIXED: Ensure float32 dtype and proper range [0, 255]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            
            # Ensure values are in [0, 255] range
            if arr.max() <= 1.0:
                arr = arr * 255.0
            
            arr = tf.image.resize(arr, [self.image_size, self.image_size])
            arr = tf.cast(arr, tf.float32)
            arr = (arr / 127.5) - 1.0  # MobileNetV2 preprocessing
            processed.append(arr.numpy())
        
        return np.array(processed, dtype=np.float32)
    
    def train_with_augmented_data(self, all_images: List, all_labels: List, validation_split: float = 0.2) -> keras.callbacks.History:
        """Train with augmented data using enhanced callbacks and strategies"""
        try:
            # CRITICAL FIX: Split data BEFORE augmentation to prevent data leakage
            from sklearn.model_selection import train_test_split
            
            # Split original images into train/validation
            train_images, val_images, train_labels, val_labels = train_test_split(
                all_images, all_labels, 
                test_size=validation_split, 
                random_state=42,
                stratify=all_labels
            )
            
            logger.info(f"Data split: {len(train_images)} train, {len(val_images)} validation")
            logger.info(f"Train labels: {np.sum(train_labels)} genuine, {len(train_labels) - np.sum(train_labels)} forged")
            logger.info(f"Val labels: {np.sum(val_labels)} genuine, {len(val_labels) - np.sum(val_labels)} forged")
            
            # Prepare training data
            train_input_a, train_input_b, train_pair_labels = self.prepare_augmented_data(train_images, train_labels)
            
            # Prepare validation data (no augmentation to prevent leakage)
            val_input_a, val_input_b, val_pair_labels = self.prepare_augmented_data(val_images, val_labels)
            
            if len(train_pair_labels) == 0:
                raise ValueError("No training pairs could be created")
            
            logger.info(f"Training with {len(train_pair_labels)} pairs ({np.sum(train_pair_labels)} positive, {len(train_pair_labels) - np.sum(train_pair_labels)} negative)")
            logger.info(f"Validation with {len(val_pair_labels)} pairs ({np.sum(val_pair_labels)} positive, {len(val_pair_labels) - np.sum(val_pair_labels)} negative)")
            
            # Create model
            self.model = self.create_siamese_network()
            
            # Enhanced callbacks
            callbacks = self._create_callbacks()
            
            # FIXED: Use separate validation data instead of validation_split
            history = self.model.fit(
                [train_input_a, train_input_b], train_pair_labels,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=([val_input_a, val_input_b], val_pair_labels),
                callbacks=callbacks,
                    verbose=1
                )
            
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _create_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create enhanced training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=15,
                restore_best_weights=True,
                mode='max',
                min_delta=0.001,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            # Track best model based on validation AUC
            keras.callbacks.ModelCheckpoint(
                filepath='best_model_checkpoint.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=0
            )
        ]
        
        return callbacks
    
    def _train_with_gradient_accumulation(self, input_a, input_b, labels, 
                                         validation_split, callbacks):
        """Custom training with gradient accumulation for CPU efficiency"""
        # Split data
        val_size = int(len(labels) * validation_split)
        val_indices = np.random.choice(len(labels), val_size, replace=False)
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[val_indices] = False
        
        train_a, train_b = input_a[train_mask], input_b[train_mask]
        train_y = labels[train_mask]
        val_a, val_b = input_a[val_indices], input_b[val_indices]
        val_y = labels[val_indices]
        
        # Effective batch size
        effective_batch = self.batch_size * self.gradient_accumulation_steps
        
        # Standard training with adjusted batch size
        history = self.model.fit(
            [train_a, train_b], train_y,
            batch_size=effective_batch,
            epochs=self.epochs,
            validation_data=([val_a, val_b], val_y),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save full model
        self.model.save(filepath, save_format='keras')
        logger.info(f"Model saved to {filepath}")
    
    def save_embedding_model(self, filepath: str):
        """Save only the embedding model for faster inference"""
        if self.embedding_model is None:
            raise ValueError("No embedding model to save")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save embedding model
        self.embedding_model.save(filepath, save_format='keras')
        logger.info(f"Embedding model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        try:
            # Load model without custom objects since we removed Lambda layers
            self.model = keras.models.load_model(filepath, compile=True, safe_mode=False)
            
            # Extract embedding model
            self._extract_embedding_model()
            
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ValueError(f"Model loading failed. This model may be from an incompatible version. Please retrain the student and try again. Error: {str(e)}")
    
    def _extract_embedding_model(self):
        """Extract embedding model from full model"""
        if self.model is None:
            return
        
        # Find the embedding branch in the model
        for layer in self.model.layers:
            if 'embedding_branch' in layer.name:
                self.embedding_model = layer
                break
    
    def verify_signature(self, test_image: Union[np.ndarray, Image.Image], 
                        centroid: np.ndarray, threshold: float) -> Tuple[bool, float, float]:
        """Verify a signature against stored prototype
        
        Returns:
            Tuple of (is_genuine, distance, confidence_score)
        """
        # Get embedding for test image
        test_embedding = self.embed_images([test_image])[0]
        
        # Calculate distance to centroid
        distance = float(np.linalg.norm(test_embedding - centroid))
        
        # Determine if genuine
        is_genuine = distance <= threshold
        
        # Calculate confidence score (0-1, higher is more confident)
        if threshold > 0:
            # Normalize distance to confidence
            confidence = max(0, 1 - (distance / threshold))
        else:
            confidence = 0.5
        
        return is_genuine, distance, confidence
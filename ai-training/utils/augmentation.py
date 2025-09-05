# augmentation.py - COMPLETE ENHANCED VERSION
import numpy as np
import cv2
from typing import List, Tuple, Optional
import random
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SignatureAugmentation:
    """Enhanced augmentation pipeline for better generalization"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 brightness_range: float = 0.2,
                 blur_probability: float = 0.3,
                 thickness_variation: float = 0.1,
                 elastic_alpha: float = 8.0,
                 elastic_sigma: float = 4.0,
                 noise_stddev: float = 6.0,
                 shear_range: float = 0.2,
                 perspective_distortion: float = 0.02):
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.blur_probability = blur_probability
        self.thickness_variation = thickness_variation
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.noise_stddev = noise_stddev
        self.shear_range = shear_range
        self.perspective_distortion = perspective_distortion
        
        # Progressive difficulty settings
        self.difficulty_levels = {
            'easy': 0.3,
            'medium': 0.6,
            'hard': 0.9
        }
    
    def augment_image(self, image: np.ndarray, is_genuine: bool = True, 
                     difficulty: str = 'medium') -> np.ndarray:
        """Apply augmentations with different strategies for genuine vs forged"""
        augmented = self._ensure_uint8(image)
        
        # Get difficulty multiplier
        difficulty_mult = self.difficulty_levels.get(difficulty, 0.6)
        
        if is_genuine:
            # Aggressive augmentation for genuine signatures
            augmentation_pipeline = [
                (0.9 * difficulty_mult, self._apply_rotation),
                (0.8 * difficulty_mult, self._apply_scale),
                (0.7 * difficulty_mult, self._apply_brightness),
                (0.6 * difficulty_mult, self._apply_elastic_distortion),
                (0.5 * difficulty_mult, self._apply_thickness_variation),
                (0.4 * difficulty_mult, self._apply_shear),
                (0.4 * difficulty_mult, self._apply_perspective),
                (self.blur_probability * difficulty_mult, self._apply_blur),
                (0.3 * difficulty_mult, self._apply_noise),
                (0.2 * difficulty_mult, self._apply_pen_pressure_variation),
            ]
        else:
            # Moderate augmentation for forged signatures
            augmentation_pipeline = [
                (0.5 * difficulty_mult, self._apply_rotation),
                (0.4 * difficulty_mult, self._apply_scale),
                (0.3 * difficulty_mult, self._apply_brightness),
                (0.3 * difficulty_mult, self._apply_elastic_distortion),
                (0.2 * difficulty_mult, self._apply_thickness_variation),
                (0.2 * difficulty_mult, self._apply_shear),
                (self.blur_probability * 0.5 * difficulty_mult, self._apply_blur),
                (0.2 * difficulty_mult, self._apply_noise),
            ]
        
        for probability, augment_func in augmentation_pipeline:
            if random.random() < probability:
                try:
                    augmented = augment_func(augmented)
                    augmented = self._ensure_uint8(augmented)
                except Exception as e:
                    logger.debug(f"Augmentation {augment_func.__name__} failed: {e}")
                    continue
        
        return augmented
    
    def _ensure_uint8(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is uint8 format"""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                return (image * 255).astype(np.uint8)
            else:
                return np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    def _apply_shear(self, image: np.ndarray) -> np.ndarray:
        """Apply shear transformation"""
        shear_x = random.uniform(-self.shear_range, self.shear_range)
        shear_y = random.uniform(-self.shear_range * 0.5, self.shear_range * 0.5)
        
        h, w = image.shape[:2]
        shear_matrix = np.array([
            [1, shear_x, -shear_x * w / 2],
            [shear_y, 1, -shear_y * h / 2]
        ], dtype=np.float32)
        
        return cv2.warpAffine(image, shear_matrix, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    
    def _apply_perspective(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective transformation"""
        h, w = image.shape[:2]
        
        # Define source points (corners)
        src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        
        # Add random distortion to corners
        dst_pts = src_pts.copy()
        for i in range(4):
            dst_pts[i][0] += random.uniform(-w * self.perspective_distortion, 
                                           w * self.perspective_distortion)
            dst_pts[i][1] += random.uniform(-h * self.perspective_distortion,
                                           h * self.perspective_distortion)
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, matrix, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
    
    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """Apply rotation with proper boundary handling"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix with proper scaling to avoid cropping
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new boundaries
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=255)
        
        # Resize back to original size
        return cv2.resize(rotated, (w, h), interpolation=cv2.INTER_AREA)
    
    def _apply_scale(self, image: np.ndarray) -> np.ndarray:
        """Apply scaling with proper interpolation"""
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use appropriate interpolation
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        if scale > 1.0:
            # Center crop
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad with white
            result = np.full((h, w), 255, dtype=np.uint8)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            result[start_h:start_h + new_h, start_w:start_w + new_w] = resized
            return result
    
    def _apply_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness with gamma correction"""
        factor = random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
        
        # Use gamma correction for more realistic brightness changes
        gamma = 1.0 / factor if factor > 0 else 1.0
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        return cv2.LUT(image, table)
    
    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply various blur types"""
        blur_type = random.choice(['gaussian', 'motion', 'median'])
        
        if blur_type == 'gaussian':
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.5, 2.0)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif blur_type == 'motion':
            kernel_size = random.choice([5, 7, 9])
            angle = random.uniform(0, 180)
            return self._motion_blur(image, kernel_size, angle)
        else:  # median
            kernel_size = random.choice([3, 5])
            return cv2.medianBlur(image, kernel_size)
    
    def _motion_blur(self, image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        """Apply directional motion blur"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        return cv2.filter2D(image, -1, kernel)
    
    def _apply_thickness_variation(self, image: np.ndarray) -> np.ndarray:
        """Vary stroke thickness with morphological operations"""
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        if random.random() < 0.5:
            # Thicken strokes
            kernel_size = random.choice([2, 3])
            iterations = random.randint(1, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary = cv2.dilate(binary, kernel, iterations=iterations)
        else:
            # Thin strokes
            kernel_size = 2
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            binary = cv2.erode(binary, kernel, iterations=1)
        
        return binary
    
    def _apply_elastic_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic distortion for natural variability"""
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1), 
            (17, 17), 
            self.elastic_sigma
        ) * self.elastic_alpha
        
        dy = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1), 
            (17, 17), 
            self.elastic_sigma
        ) * self.elastic_alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, 
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    
    def _apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic noise patterns"""
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_stddev, image.shape)
            noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_pepper':
            noisy = image.copy()
            prob = 0.01
            mask = np.random.random(image.shape) < prob
            noisy[mask] = random.choice([0, 255])
        else:  # speckle
            noise = np.random.randn(*image.shape) * 0.1
            noisy = np.clip(image + image * noise, 0, 255).astype(np.uint8)
        
        return noisy
    
    def _apply_pen_pressure_variation(self, image: np.ndarray) -> np.ndarray:
        """Simulate natural pen pressure variations"""
        # Create pressure map
        h, w = image.shape[:2]
        
        # Generate smooth random pressure field
        pressure_map = np.random.rand(h // 10, w // 10)
        pressure_map = cv2.resize(pressure_map, (w, h), interpolation=cv2.INTER_CUBIC)
        pressure_map = cv2.GaussianBlur(pressure_map, (21, 21), 5)
        
        # Normalize between 0.7 and 1.0 (simulate pressure range)
        pressure_map = 0.7 + 0.3 * pressure_map
        
        # Apply pressure to image intensity
        result = image.astype(np.float32)
        result = result * pressure_map
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def augment_batch(self, images: List[np.ndarray], labels: List[bool], 
                     augmentation_factor: int = 3,
                     progressive_difficulty: bool = True) -> Tuple[List[np.ndarray], List[bool]]:
        """Augment batch with progressive difficulty"""
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Always include original
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Create augmented versions with increasing difficulty
            for i in range(augmentation_factor):
                if progressive_difficulty:
                    # Progressively increase augmentation intensity
                    difficulty = ['easy', 'medium', 'hard'][min(i, 2)]
                else:
                    difficulty = 'medium'
                
                aug_img = self.augment_image(image, is_genuine=label, difficulty=difficulty)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return augmented_images, augmented_labels
    
    def create_test_time_augmentation(self, image: np.ndarray, num_variants: int = 5) -> List[np.ndarray]:
        """Create mild augmentations for test-time averaging"""
        variants = [image]  # Include original
        
        for _ in range(num_variants - 1):
            # Use very mild augmentations for TTA
            mild_aug = self._apply_mild_augmentation(image)
            variants.append(mild_aug)
        
        return variants
    
    def _apply_mild_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply mild augmentation for test-time"""
        augmented = self._ensure_uint8(image)
        
        # Very mild transformations
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)  # Smaller rotation
            augmented = self._apply_custom_rotation(augmented, angle)
        
        if random.random() < 0.5:
            scale = random.uniform(0.95, 1.05)  # Smaller scale
            augmented = self._apply_custom_scale(augmented, scale)
        
        if random.random() < 0.3:
            augmented = self._apply_mild_brightness(augmented)
        
        return augmented
    
    def _apply_custom_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Apply specific rotation angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    
    def _apply_custom_scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Apply specific scale factor"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Always center the result
        if scale != 1.0:
            result = np.full((h, w), 255, dtype=np.uint8)
            y_offset = abs(h - new_h) // 2
            x_offset = abs(w - new_w) // 2
            
            if scale > 1.0:
                result = resized[y_offset:y_offset+h, x_offset:x_offset+w]
            else:
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return result
        
        return resized
    
    def _apply_mild_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply mild brightness adjustment"""
        factor = random.uniform(0.95, 1.05)
        return np.clip(image * factor, 0, 255).astype(np.uint8)
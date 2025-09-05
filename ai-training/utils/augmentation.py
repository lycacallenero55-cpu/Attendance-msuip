import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
import random
import logging

logger = logging.getLogger(__name__)


class SignatureAugmentation:
    """Enhanced data augmentation specifically for signature images"""

    def __init__(self,
                 rotation_range=15.0,
                 scale_range=(0.9, 1.1),
                 brightness_range=0.2,
                 blur_probability=0.3,
                 thickness_variation=0.15,
                 noise_probability=0.2,
                 shear_range=0.1):

        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.blur_probability = blur_probability
        self.thickness_variation = thickness_variation
        self.noise_probability = noise_probability
        self.shear_range = shear_range

    def augment_single(self, image, is_genuine=True):
        """Apply augmentation to a single image - FIXED to ensure consistent output"""
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Store original size
        original_size = image.size

        # Apply augmentations
        augment_prob = 0.8 if is_genuine else 0.6

        if random.random() < augment_prob:
            # Rotation
            if random.random() < 0.7:
                angle = random.uniform(-self.rotation_range, self.rotation_range)
                image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)

            # Scaling - FIXED to maintain size
            if random.random() < 0.6:
                scale = random.uniform(*self.scale_range)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Always resize back to original size
                image = image.resize(original_size, Image.Resampling.LANCZOS)

            # Brightness variation
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(image)
                factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
                image = enhancer.enhance(factor)

            # Blur
            if random.random() < self.blur_probability:
                blur_type = random.choice(['gaussian', 'motion'])
                if blur_type == 'gaussian':
                    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
                else:
                    image = self._apply_motion_blur(image)

            # Thickness variation
            if random.random() < 0.5:
                image = self._vary_stroke_thickness(image, self.thickness_variation)

            # Add noise
            if random.random() < self.noise_probability:
                image = self._add_noise(image)

            # Shear
            if random.random() < 0.4:
                image = self._apply_shear(image, self.shear_range)

        # CRITICAL: Ensure final image has consistent size
        if image.size != original_size:
            image = image.resize(original_size, Image.Resampling.LANCZOS)

        return image

    def _apply_motion_blur(self, image):
        """Apply motion blur to simulate pen movement"""
        # Convert to numpy
        img_array = np.array(image)

        # Random kernel size and angle
        kernel_size = random.choice([3, 5, 7])
        angle = random.uniform(0, 180)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1

        # Rotate kernel
        M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)

        # Apply blur
        blurred = cv2.filter2D(img_array, -1, kernel)

        return Image.fromarray(blurred)

    def _vary_stroke_thickness(self, image, variation):
        """Vary stroke thickness using morphological operations"""
        # Convert to grayscale for processing
        gray = image.convert('L')
        img_array = np.array(gray)

        # Invert (signatures are usually dark on light)
        img_array = 255 - img_array

        # Random morphological operation
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if random.random() < 0.5:
            # Dilation (thicker strokes)
            processed = cv2.dilate(img_array, kernel, iterations=1)
        else:
            # Erosion (thinner strokes)
            processed = cv2.erode(img_array, kernel, iterations=1)

        # Invert back
        processed = 255 - processed

        # Convert back to RGB
        result = Image.fromarray(processed).convert('RGB')

        # Blend with original
        blend_factor = variation
        result = Image.blend(image, result, blend_factor)

        return result

    def _add_noise(self, image):
        """Add realistic noise to signature"""
        img_array = np.array(image).astype(np.float32)

        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])

        if noise_type == 'gaussian':
            # Gaussian noise
            noise = np.random.randn(*img_array.shape) * random.uniform(5, 15)
            img_array = img_array + noise

        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            prob = random.uniform(0.01, 0.03)
            mask = np.random.random(img_array.shape[:2])
            img_array[mask < prob / 2] = 0
            img_array[mask > 1 - prob / 2] = 255

        else:  # speckle
            # Speckle noise
            noise = np.random.randn(*img_array.shape) * 0.1
            img_array = img_array + img_array * noise

        # Clip values
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def _apply_shear(self, image, shear_range):
        """Apply shear transformation"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Random shear values
        shear_x = random.uniform(-shear_range, shear_range)
        shear_y = random.uniform(-shear_range, shear_range)

        # Shear matrix
        M = np.array([
            [1, shear_x, -shear_x * height / 2],
            [shear_y, 1, -shear_y * width / 2]
        ], dtype=np.float32)

        # Apply transformation
        sheared = cv2.warpAffine(img_array, M, (width, height),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

        return Image.fromarray(sheared)

    def augment_batch(self, images, labels, augmentation_factor=2):
        """Augment a batch of images - FIXED to ensure consistent output format"""
        # Ensure augmentation_factor is integer
        try:
            if augmentation_factor is None:
                augmentation_factor = 2
            augmentation_factor = int(round(augmentation_factor))
            if augmentation_factor < 1:
                augmentation_factor = 1
        except (TypeError, ValueError):
            logger.warning(f"Invalid augmentation_factor: {augmentation_factor}, using default of 2")
            augmentation_factor = 2

        augmented_images = []
        augmented_labels = []

        # Determine target size from first image
        target_size = None
        if images and len(images) > 0:
            first_img = images[0]
            if isinstance(first_img, Image.Image):
                target_size = first_img.size
            elif isinstance(first_img, np.ndarray):
                target_size = (first_img.shape[1], first_img.shape[0])  # (width, height)

        for img, label in zip(images, labels):
            # Ensure consistent size for original
            if isinstance(img, Image.Image):
                if target_size and img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Always include original
            augmented_images.append(img)
            augmented_labels.append(label)

            # Generate augmented versions
            num_augmentations = max(0, augmentation_factor - 1)
            for _ in range(num_augmentations):
                aug_img = self.augment_single(img, is_genuine=label)
                # Ensure augmented image has the target size
                if target_size and isinstance(aug_img, Image.Image):
                    if aug_img.size != target_size:
                        aug_img = aug_img.resize(target_size, Image.Resampling.LANCZOS)
                augmented_images.append(aug_img)
                augmented_labels.append(label)

        logger.info(f"Augmented {len(images)} images to {len(augmented_images)} samples (factor: {augmentation_factor})")

        return augmented_images, augmented_labels

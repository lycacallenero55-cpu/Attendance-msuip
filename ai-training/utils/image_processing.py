from PIL import Image
import numpy as np
import cv2
import io
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image, target_size, enhance=True):
    """Enhanced preprocessing for signature images - FIXED for consistent output"""
    try:
        # Handle different input types
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Handle PNG/transparency
        if image.format == 'PNG' or hasattr(image, 'mode'):
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                if len(image.split()) > 3:
                    background.paste(image, mask=image.split()[3])
                else:
                    background.paste(image)
                image = background
            elif image.mode == 'LA' or image.mode == 'L':
                image = image.convert('RGB')
            elif image.mode == 'P':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
        
        # Optional enhancement
        if enhance:
            image = enhance_signature_image(image)
        
        # Smart resize maintaining aspect ratio
        image = smart_resize(image, target_size)
        
        # CRITICAL: Ensure final size is exactly target_size x target_size
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return image
    
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # Return a blank image of the correct size as fallback
        return Image.new('RGB', (target_size, target_size), (255, 255, 255))


def smart_resize(image, target_size):
    """Resize image maintaining aspect ratio with padding"""
    # Get current size
    width, height = image.size
    
    # Calculate scaling factor
    scale = min(target_size / width, target_size / height)
    
    # Calculate new size
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    # Calculate padding
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    
    # Paste resized image
    padded.paste(image, (left, top))
    
    return padded


def enhance_signature_image(image):
    """Enhance signature image for better feature extraction"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Apply slight Gaussian blur to smooth edges
        smoothed = cv2.GaussianBlur(denoised, (3, 3), 0.5)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2RGB)
        
        # Enhance contrast slightly
        alpha = 1.1  # Contrast control
        beta = -10   # Brightness control
        enhanced_rgb = cv2.convertScaleAbs(enhanced_rgb, alpha=alpha, beta=beta)
        
        return Image.fromarray(enhanced_rgb)
    
    except Exception as e:
        logger.warning(f"Image enhancement failed, using original: {e}")
        return image


def get_image_quality_score(image):
    """Calculate comprehensive image quality score"""
    try:
        # Convert to grayscale for analysis
        if isinstance(image, Image.Image):
            gray = np.array(image.convert('L'))
        else:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Multiple quality metrics
        metrics = {}
        
        # 1. Variance (sharpness/detail)
        metrics['variance'] = np.var(gray)
        
        # 2. Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['focus'] = laplacian.var()
        
        # 3. Edge density (signature structure)
        edges = cv2.Canny(gray, 50, 150)
        metrics['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 4. Contrast (dynamic range)
        metrics['contrast'] = gray.std()
        
        # 5. Entropy (information content)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros for log
        metrics['entropy'] = -np.sum(hist * np.log2(hist))
        
        # Normalize and combine metrics
        weights = {
            'variance': 0.2,
            'focus': 0.3,
            'edge_density': 0.2,
            'contrast': 0.15,
            'entropy': 0.15
        }
        
        # Normalize each metric to 0-1 range
        normalized_scores = {}
        normalized_scores['variance'] = min(metrics['variance'] / 5000, 1.0)
        normalized_scores['focus'] = min(metrics['focus'] / 1000, 1.0)
        normalized_scores['edge_density'] = min(metrics['edge_density'] * 10, 1.0)
        normalized_scores['contrast'] = min(metrics['contrast'] / 100, 1.0)
        normalized_scores['entropy'] = metrics['entropy'] / 8.0  # Max entropy is 8 for 256 bins
        
        # Weighted average
        quality_score = sum(normalized_scores[key] * weights[key] for key in weights)
        
        return float(min(quality_score, 1.0))
    
    except Exception as e:
        logger.error(f"Quality score calculation failed: {e}")
        return 0.5  # Default middle score on error


def validate_signature_image(image, min_quality_score=0.3):
    """Validate if image is suitable for signature verification"""
    try:
        # Get quality score
        quality = get_image_quality_score(image)
        
        # Check minimum quality
        if quality < min_quality_score:
            return False, f"Image quality too low: {quality:.2f}"
        
        # Check image dimensions
        width, height = image.size if hasattr(image, 'size') else image.shape[:2]
        if width < 50 or height < 50:
            return False, "Image too small (minimum 50x50)"
        
        if width > 4000 or height > 4000:
            return False, "Image too large (maximum 4000x4000)"
        
        # Check if mostly blank
        gray = np.array(image.convert('L') if hasattr(image, 'convert') else image)
        non_white_ratio = np.sum(gray < 250) / gray.size
        if non_white_ratio < 0.01:
            return False, "Image appears to be blank"
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"
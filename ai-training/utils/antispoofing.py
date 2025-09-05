import numpy as np
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class AntiSpoofingDetector:
    """Detect potentially spoofed or printed signatures"""
    
    def __init__(self):
        self.printed_threshold = 0.6
        self.quality_threshold = 0.3
    
    def analyze_signature(self, image):
        """Analyze signature for spoofing indicators"""
        try:
            # CRITICAL FIX: Convert PIL Image to numpy array first
            if isinstance(image, Image.Image):
                # Convert PIL to numpy array
                img_array = np.array(image)
                if img_array.ndim == 2:  # Grayscale
                    gray = img_array
                else:  # RGB
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                # Already numpy array
                img_array = np.array(image)
                if img_array.ndim == 2:
                    gray = img_array
                else:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Ensure uint8 type for OpenCV
            if gray.dtype != np.uint8:
                if gray.max() <= 1.0:
                    gray = (gray * 255).astype(np.uint8)
                else:
                    gray = gray.astype(np.uint8)
            
            # Now OpenCV functions will work
            analysis = {
                "is_potentially_spoofed": False,
                "is_likely_printed": False,
                "is_low_quality": False,
                "printed_confidence": 0.0,
                "quality_score": 0.0,
                "details": {}
            }
            
            # Detect if signature is printed/photocopied
            printed_score = self._detect_printed_signature(gray)
            analysis["printed_confidence"] = float(printed_score)
            analysis["is_likely_printed"] = printed_score > self.printed_threshold
            
            # Assess image quality
            quality_score = self._assess_image_quality(gray)
            analysis["quality_score"] = float(quality_score)
            analysis["is_low_quality"] = quality_score < self.quality_threshold
            
            # Overall spoofing detection
            if analysis["is_likely_printed"] or analysis["is_low_quality"]:
                analysis["is_potentially_spoofed"] = True
            
            # Additional checks
            analysis["details"] = {
                "has_uniform_thickness": self._check_uniform_thickness(gray),
                "has_artificial_edges": self._check_artificial_edges(gray),
                "histogram_entropy": self._calculate_histogram_entropy(gray)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Anti-spoofing analysis failed: {e}")
            return {
                "is_potentially_spoofed": False,
                "is_likely_printed": False,
                "is_low_quality": False,
                "printed_confidence": 0.0,
                "quality_score": 0.5,
                "error": str(e)
            }
    
    def _detect_printed_signature(self, gray):
        """Detect if signature appears to be printed/photocopied"""
        try:
            # Edge detection to find sharp boundaries (common in printed images)
            edges = cv2.Canny(gray, 50, 150)
            
            # Printed signatures often have more uniform edge intensities
            edge_std = np.std(edges[edges > 0]) if np.any(edges > 0) else 0
            
            # Check for halftone patterns (common in printed material)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Look for regular patterns in frequency domain
            center = np.array(magnitude.shape) // 2
            mask = np.zeros_like(magnitude, dtype=np.uint8)  # Ensure uint8 dtype for cv2.circle
            cv2.circle(mask, tuple(center.astype(int)), int(min(center) // 4), 1, -1)
            high_freq_energy = np.sum(magnitude * (1 - mask))
            total_energy = np.sum(magnitude)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-6)
            
            # Binary threshold analysis
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Printed signatures have less variation in stroke darkness
            stroke_pixels = gray[binary < 128]
            stroke_variance = np.var(stroke_pixels) if len(stroke_pixels) > 0 else 0
            
            # Combine indicators
            printed_score = 0.0
            
            # Sharp edges indicator
            if edge_std < 50:
                printed_score += 0.3
            
            # High frequency patterns (halftone)
            if high_freq_ratio > 0.3:
                printed_score += 0.3
            
            # Low stroke variance (uniform ink)
            if stroke_variance < 500:
                printed_score += 0.4
            
            return min(1.0, printed_score)
            
        except Exception as e:
            logger.error(f"Error detecting printed signature: {e}")
            return 0.0
    
    def _assess_image_quality(self, gray):
        """Assess overall image quality"""
        try:
            # Laplacian variance (focus measure)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_measure = laplacian.var()
            
            # Contrast measure
            contrast = gray.std()
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Normalize and combine
            quality_score = 0.0
            
            # Good focus
            if focus_measure > 100:
                quality_score += 0.4
            elif focus_measure > 50:
                quality_score += 0.2
            
            # Good contrast
            if contrast > 30:
                quality_score += 0.3
            elif contrast > 15:
                quality_score += 0.15
            
            # Appropriate edge density
            if 0.01 < edge_density < 0.3:
                quality_score += 0.3
            elif 0.005 < edge_density < 0.5:
                quality_score += 0.15
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return 0.5
    
    def _check_uniform_thickness(self, gray):
        """Check if strokes have suspiciously uniform thickness"""
        try:
            # Morphological analysis
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(gray, kernel, iterations=1)
            eroded = cv2.erode(gray, kernel, iterations=1)
            
            # Stroke width map
            stroke_width = dilated - eroded
            
            # Check variance in stroke width
            stroke_pixels = stroke_width[stroke_width > 10]
            if len(stroke_pixels) > 100:
                width_variance = np.var(stroke_pixels)
                return width_variance < 100  # Low variance = uniform
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking uniform thickness: {e}")
            return False
    
    def _check_artificial_edges(self, gray):
        """Check for artificial-looking edges"""
        try:
            # Gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Artificial edges often have very consistent gradients
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            significant_grads = grad_mag[grad_mag > 10]
            
            if len(significant_grads) > 100:
                grad_std = np.std(significant_grads)
                return grad_std < 20  # Very consistent = artificial
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking artificial edges: {e}")
            return False
    
    def _calculate_histogram_entropy(self, gray):
        """Calculate histogram entropy (natural images have higher entropy)"""
        try:
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist = hist / (hist.sum() + 1e-6)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-6))
            return float(entropy)
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 5.0
    
    def get_spoofing_warning_message(self, analysis):
        """Generate appropriate warning message based on analysis"""
        if not analysis.get("is_potentially_spoofed"):
            return None
        
        warnings = []
        
        if analysis.get("is_likely_printed"):
            warnings.append(f"Signature appears to be printed or photocopied (confidence: {analysis['printed_confidence']:.1%})")
        
        if analysis.get("is_low_quality"):
            warnings.append(f"Image quality is too low for reliable verification (score: {analysis['quality_score']:.1%})")
        
        details = analysis.get("details", {})
        if details.get("has_uniform_thickness"):
            warnings.append("Stroke thickness appears artificially uniform")
        
        if details.get("has_artificial_edges"):
            warnings.append("Edge patterns suggest digital manipulation")
        
        return " | ".join(warnings) if warnings else None
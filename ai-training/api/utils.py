from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/preview")
async def generate_image_preview(file: UploadFile = File(...)):
    """Generate a PNG preview for any image format (e.g., TIFF) for browser display."""
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data))
        # Convert to RGB for consistent PNG encoding
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        # Optionally limit preview size
        max_side = 1600
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to generate preview for image")
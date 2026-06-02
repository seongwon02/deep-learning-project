"""
Face cropping and validation using InsightFace.
Validates detected faces to filter out false positives and extracts clean facial clips.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image, ImageDraw, ImageFont

class InsightFaceValidator:
    def __init__(self, models_dir: Path):
        from insightface.app import FaceAnalysis
        # buffalo_l is used for high-fidelity cross-validation
        self.app = FaceAnalysis(name="buffalo_l", root=str(models_dir))
        self.app.prepare(ctx_id=-1, det_size=(640, 640)) # Run on CPU

    def validate(self, crop_img: Image.Image) -> bool:
        rgb = np.asarray(crop_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            return False
        best_face = max(faces, key=lambda f: float(f.det_score))
        return float(best_face.det_score) > 0.40

def crop_and_validate_faces(
    input_path: str | Path,
    detections: Dict[str, Any] | List[Dict[str, Any]],
    output_dir: str | Path,
    pad: float = 0.35,
    cross_validate: bool = True
) -> Dict[str, Any]:
    """
    Crops detected faces from the input video/image and optionally validates them via InsightFace.
    Saves cropped faces inside output_dir and returns metadata.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standardize detection format
    by_frame = {}
    if isinstance(detections, dict) and "frames" in detections:
        for item in detections["frames"]:
            fidx = item.get("frame_index", 0)
            faces = item.get("faces", [])
            by_frame[fidx] = faces
    elif isinstance(detections, list):
        by_frame[0] = detections
    else:
        # Assuming dictionary mapping key frames
        for k, v in detections.items():
            clean_key = "".join(c for c in k if c.isdigit())
            fidx = int(clean_key) if clean_key else 0
            if isinstance(v, list):
                by_frame[fidx] = v
                
    # Find the first frame index where each unique track ID appears
    id_to_first_frame = {}
    for frame_idx, faces in sorted(by_frame.items()):
        for face in faces:
            tid = str(face.get("track_id", face.get("id", "")))
            if tid and tid not in id_to_first_frame:
                bbox = face.get("bbox") or face.get("box") or face.get("xyxy")
                if bbox:
                    id_to_first_frame[tid] = (frame_idx, bbox)
                    
    # Initialize validator if cross-validation is active
    validator = None
    if cross_validate:
        curr_dir = Path(__file__).parent.resolve()
        possible_models_dirs = [
            curr_dir.parent,                                   # real/anon_project/
            curr_dir.parent.parent / "real",                   # real/
            curr_dir.parent.parent                             # project root /
        ]
        models_dir = None
        for p in possible_models_dirs:
            if (p / "models").exists():
                models_dir = p
                break
        if models_dir is None:
            models_dir = curr_dir.parent.parent # Default to project root
            
        print(f"[InsightFace Validator] Loading Buffalo model from: {models_dir / 'models'}")
        try:
            validator = InsightFaceValidator(models_dir)
        except Exception as e:
            print(f"[InsightFace Validator] Warning: Failed to load InsightFace validation model: {e}. Skipping validation.")
            cross_validate = False
            
    # Group by frame to minimize seeks
    frame_to_ids = {}
    for tid, (fidx, bbox) in id_to_first_frame.items():
        frame_to_ids.setdefault(fidx, []).append((tid, bbox))
        
    ext = input_path.suffix.lower()
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    validated_count = 0
    skipped_count = 0
    results_meta = {}
    
    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        current_frame = 0
        for fidx in sorted(frame_to_ids.keys()):
            if fidx != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                current_frame = fidx
                
            ret, frame = cap.read()
            if not ret:
                continue
            current_frame += 1
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            for tid, bbox in frame_to_ids[fidx]:
                x1, y1, x2, y2 = map(int, bbox[:4])
                w, h = x2 - x1, y2 - y1
                
                pad_w = int(w * pad)
                pad_h = int(h * pad)
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(width, x2 + pad_w)
                cy2 = min(height, y2 + pad_h)
                
                face_crop_clean = img.crop((cx1, cy1, cx2, cy2))
                
                if cross_validate and validator is not None:
                    if not validator.validate(face_crop_clean):
                        skipped_count += 1
                        continue
                        
                # Create a copy to draw bbox inside the cropped image for preview
                face_crop = face_crop_clean.copy()
                draw = ImageDraw.Draw(face_crop)
                rx1 = x1 - cx1
                ry1 = y1 - cy1
                rx2 = x2 - cx1
                ry2 = y2 - cy1
                draw.rectangle([rx1, ry1, rx2, ry2], outline=(34, 197, 94), width=2)
                
                crop_path = output_dir / f"face_id_{tid}.jpg"
                face_crop.save(crop_path, quality=92)
                
                # Save full-frame overview for context
                overview_path = output_dir / f"overview_id_{tid}.jpg"
                overview_img = img.copy()
                ov_draw = ImageDraw.Draw(overview_img)
                ov_draw.rectangle([x1, y1, x2, y2], outline=(239, 68, 68), width=3)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                except Exception:
                    font = ImageFont.load_default()
                ov_draw.text((x1, max(0, y1 - 20)), f"ID {tid}", fill=(239, 68, 68), font=font)
                overview_img.save(overview_path, quality=85)
                
                results_meta[tid] = {
                    "crop_path": str(crop_path),
                    "overview_path": str(overview_path),
                    "frame_index": fidx,
                    "bbox": [x1, y1, x2, y2]
                }
                validated_count += 1
        cap.release()
    else:
        # Single Image
        img = Image.open(input_path).convert("RGB")
        width, height = img.size
        for tid, (_, bbox) in id_to_first_frame.items():
            x1, y1, x2, y2 = map(int, bbox[:4])
            w, h = x2 - x1, y2 - y1
            
            pad_w = int(w * pad)
            pad_h = int(h * pad)
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(width, x2 + pad_w)
            cy2 = min(height, y2 + pad_h)
            
            face_crop_clean = img.crop((cx1, cy1, cx2, cy2))
            
            if cross_validate and validator is not None:
                if not validator.validate(face_crop_clean):
                    skipped_count += 1
                    continue
                    
            face_crop = face_crop_clean.copy()
            draw = ImageDraw.Draw(face_crop)
            rx1 = x1 - cx1
            ry1 = y1 - cy1
            rx2 = x2 - cx1
            ry2 = y2 - cy1
            draw.rectangle([rx1, ry1, rx2, ry2], outline=(34, 197, 94), width=2)
            
            crop_path = output_dir / f"face_id_{tid}.jpg"
            face_crop.save(crop_path, quality=92)
            
            overview_path = output_dir / f"overview_id_{tid}.jpg"
            overview_img = img.copy()
            ov_draw = ImageDraw.Draw(overview_img)
            ov_draw.rectangle([x1, y1, x2, y2], outline=(239, 68, 68), width=3)
            overview_img.save(overview_path, quality=85)
            
            results_meta[tid] = {
                "crop_path": str(crop_path),
                "overview_path": str(overview_path),
                "frame_index": 0,
                "bbox": [x1, y1, x2, y2]
            }
            validated_count += 1
            
    print(f"[Face Cropper] Extracted {validated_count} faces. Screened out {skipped_count} false positives.")
    return {
        "validated_count": validated_count,
        "skipped_count": skipped_count,
        "faces": results_meta
    }

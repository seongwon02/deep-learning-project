"""
Gaussian blur and Pixelation anonymizer processor.
Provides clean APIs to anonymize faces in videos and images.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Optional
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageFont

def get_bbox(face: dict) -> tuple[int, int, int, int] | None:
    raw = face.get("bbox") or face.get("box") or face.get("xyxy")
    if raw is None:
        if all(k in face for k in ("x1", "y1", "x2", "y2")):
            return int(face["x1"]), int(face["y1"]), int(face["x2"]), int(face["y2"])
        if all(k in face for k in ("x", "y", "w", "h")):
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            return int(x), int(y), int(x + w), int(y + h)
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
    return None

def get_track_id(face: dict) -> str | None:
    for key in ("track_id", "trackId", "id", "track"):
        if key in face and face[key] is not None:
            return str(face[key])
    return None

def load_detections(det_path: Path) -> dict[int, list]:
    with open(det_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    by_frame = {}
    if isinstance(data, dict) and "frames" in data:
        for item in data["frames"]:
            fidx = item.get("frame_index", 0)
            faces = item.get("faces", [])
            by_frame[fidx] = faces
    elif isinstance(data, dict):
        for k, v in data.items():
            clean_key = "".join(c for c in k if c.isdigit())
            fidx = int(clean_key) if clean_key else 0
            if isinstance(v, list):
                by_frame[fidx] = v
    elif isinstance(data, list):
        by_frame[0] = data
    return by_frame

def apply_lookahead(by_frame: dict[int, list], lookahead_frames: int) -> dict[int, list]:
    if lookahead_frames <= 0:
        return by_frame
    
    track_first_occurrence = {}
    for frame_idx, faces in sorted(by_frame.items()):
        for face in faces:
            tid = get_track_id(face)
            if tid is not None and tid not in track_first_occurrence:
                bbox = get_bbox(face)
                if bbox:
                    track_first_occurrence[tid] = (frame_idx, face)
                    
    for tid, (first_fidx, face_data) in track_first_occurrence.items():
        bbox = get_bbox(face_data)
        if not bbox:
            continue
        for fidx in range(max(0, first_fidx - lookahead_frames), first_fidx):
            by_frame.setdefault(fidx, [])
            already_exists = any(get_track_id(f) == tid for f in by_frame[fidx])
            if not already_exists:
                virtual_face = face_data.copy()
                by_frame[fidx].append(virtual_face)
                
    return by_frame

def apply_blur(img: Image.Image, box: tuple[int, int, int, int], radius: float) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return img
    
    crop = img.crop((x1, y1, x2, y2))
    blurred = crop.filter(ImageFilter.GaussianBlur(radius=radius))
    
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, w, h), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(3.0, radius / 3.0)))
    
    blended = Image.composite(blurred, crop, mask)
    out = img.copy()
    out.paste(blended, (x1, y1))
    return out

def apply_pixelate(img: Image.Image, box: tuple[int, int, int, int], pixel_size: int) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return img
    
    crop = img.crop((x1, y1, x2, y2))
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    
    small = crop.resize((small_w, small_h), Image.Resampling.BILINEAR)
    pixelated = small.resize((w, h), Image.Resampling.NEAREST)
    
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, w, h), fill=255)
    
    blended = Image.composite(pixelated, crop, mask)
    out = img.copy()
    out.paste(blended, (x1, y1))
    return out

def draw_bbox_overlay(img: Image.Image, box: tuple[int, int, int, int], tid: str) -> Image.Image:
    x1, y1, x2, y2 = box
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=3)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x1, max(0, y1 - 20)), f"ID {tid}", fill=(34, 197, 94), font=font)
    return out

def draw_filled_bbox(img: Image.Image, box: tuple[int, int, int, int], tid: str) -> Image.Image:
    x1, y1, x2, y2 = box
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0), outline=(0, 0, 0))
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x1, max(0, y1 - 20)), f"ID {tid}", fill=(34, 197, 94), font=font)
    return out

def apply_blur_anonymization(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    detections: Union[Dict[str, Any], Path, str],
    keep_track_ids: Union[Set[str], List[str], str] = set(),
    fallback_mode: str = "blur",
    fallback_blur_radius: float = 18.0,
    fallback_pixel_size: int = 8,
    max_frames: int = 0,
    bbox_smoothing_alpha: float = 0.4,
    lookahead_frames: int = 0
) -> None:
    """
    Main function to run blur / pixelate anonymization.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Standardize keep_track_ids
    if isinstance(keep_track_ids, str):
        keep_ids = {x.strip() for x in keep_track_ids.split(",") if x.strip()}
    else:
        keep_ids = set(str(x) for x in keep_track_ids)
        
    # Load detections
    if isinstance(detections, (str, Path)):
        by_frame = load_detections(Path(detections))
    else:
        # Dictionary format
        by_frame = {}
        if isinstance(detections, dict) and "frames" in detections:
            for item in detections["frames"]:
                fidx = item.get("frame_index", 0)
                faces = item.get("faces", [])
                by_frame[fidx] = faces
        else:
            by_frame = detections
            
    if lookahead_frames > 0:
        by_frame = apply_lookahead(by_frame, lookahead_frames)
        
    ext = input_path.suffix.lower()
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    if not is_video:
        # Process Single Image
        img = Image.open(input_path).convert("RGB")
        width, height = img.size
        faces = by_frame.get(0, [])
        for face in faces:
            tid = get_track_id(face)
            if tid in keep_ids:
                continue
            bbox = get_bbox(face)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if fallback_mode == "blur":
                img = apply_blur(img, (x1, y1, x2, y2), fallback_blur_radius)
            elif fallback_mode == "pixelate":
                img = apply_pixelate(img, (x1, y1, x2, y2), fallback_pixel_size)
            elif fallback_mode == "bbox":
                img = draw_bbox_overlay(img, (x1, y1, x2, y2), tid)
            elif fallback_mode == "filled":
                img = draw_filled_bbox(img, (x1, y1, x2, y2), tid)
        img.save(output_path, quality=95)
        print(f"[Blur Anonymizer] Processed image saved to {output_path}")
    else:
        # Process Video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = frame_count if max_frames <= 0 else min(frame_count, max_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Could not create output video writer: {output_path}")
            
        smoothed_bboxes = {}
        frame_idx = 0
        
        print(f"[Blur Anonymizer] Anonymizing video ({total} frames)...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames > 0 and frame_idx >= max_frames):
                    break
                    
                faces = by_frame.get(frame_idx, [])
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                for face in faces:
                    tid = get_track_id(face)
                    if tid in keep_ids:
                        continue
                    bbox = get_bbox(face)
                    if not bbox:
                        continue
                        
                    # Apply EMA bbox smoothing
                    if tid is not None:
                        if tid in smoothed_bboxes:
                            prev = smoothed_bboxes[tid]
                            alpha = bbox_smoothing_alpha
                            bbox = (
                                int(alpha * bbox[0] + (1 - alpha) * prev[0]),
                                int(alpha * bbox[1] + (1 - alpha) * prev[1]),
                                int(alpha * bbox[2] + (1 - alpha) * prev[2]),
                                int(alpha * bbox[3] + (1 - alpha) * prev[3]),
                            )
                        smoothed_bboxes[tid] = bbox
                        
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    if fallback_mode == "blur":
                        img = apply_blur(img, (x1, y1, x2, y2), fallback_blur_radius)
                    elif fallback_mode == "pixelate":
                        img = apply_pixelate(img, (x1, y1, x2, y2), fallback_pixel_size)
                    elif fallback_mode == "bbox":
                        img = draw_bbox_overlay(img, (x1, y1, x2, y2), tid)
                    elif fallback_mode == "filled":
                        img = draw_filled_bbox(img, (x1, y1, x2, y2), tid)
                        
                frame_out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                writer.write(frame_out)
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"[Blur Anonymizer] Processed {frame_idx}/{total} frames")
        finally:
            cap.release()
            writer.release()
        print(f"[Blur Anonymizer] Processed video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blur/Pixelate CLI wrapper")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--keep-track-ids", type=str, default="")
    parser.add_argument("--fallback-mode", choices=("blur", "pixelate", "bbox", "filled"), default="blur")
    parser.add_argument("--fallback-blur-radius", type=float, default=18.0)
    parser.add_argument("--fallback-pixel-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--bbox-smoothing-alpha", type=float, default=0.4)
    parser.add_argument("--lookahead-frames", type=int, default=0)
    args = parser.parse_args()
    
    apply_blur_anonymization(
        args.input, args.output, args.detections,
        keep_track_ids=args.keep_track_ids,
        fallback_mode=args.fallback_mode,
        fallback_blur_radius=args.fallback_blur_radius,
        fallback_pixel_size=args.fallback_pixel_size,
        max_frames=args.max_frames,
        bbox_smoothing_alpha=args.bbox_smoothing_alpha,
        lookahead_frames=args.lookahead_frames
    )

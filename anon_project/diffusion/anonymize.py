#!/usr/bin/env python3
"""Selective face anonymization with YOLO track IDs and SDXL inpainting."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from owner_reid import (
    InsightFaceEmbedder,
    OwnerMatcher,
    OwnerProfile,
    build_owner_profile,
    export_face_crops,
)
from reference_face_conditioning import ReferenceFaceBank
from reference_identity_router import (
    REFERENCE_ROUTE_INSTANTID,
    REFERENCE_ROUTE_IP_ADAPTER,
    InsightFaceReferenceAnalyzer,
    ReferenceIdentityCondition,
    draw_landmark_condition,
    route_reference_identity,
)
from reference_prompt import (
    DEFAULT_CAPTION_MODEL,
    build_reference_prompt_from_paths,
    load_reference_prompt,
    write_reference_prompt,
)


# Monkey patch diffusers to avoid IndexError when loading LoRAs for SDXL text encoder
try:
    import diffusers.loaders.lora_base
    import diffusers.loaders.lora_pipeline
    original_load_lora_into_text_encoder = diffusers.loaders.lora_base._load_lora_into_text_encoder

    def patched_load_lora_into_text_encoder(
        state_dict,
        network_alphas,
        text_encoder,
        prefix=None,
        lora_scale=1.0,
        text_encoder_name="text_encoder",
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
    ):
        new_state_dict = {}
        prefix_to_check = f"{prefix}." if prefix is not None else ""
        target_prefix = f"{prefix_to_check}text_model.encoder."
        replacement_prefix = f"{prefix_to_check}encoder."
        
        if not hasattr(text_encoder, "text_model"):
            for k, v in state_dict.items():
                if k.startswith(target_prefix):
                    new_key = k.replace(target_prefix, replacement_prefix, 1)
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            
            if network_alphas is not None:
                new_alphas = {}
                for k, v in network_alphas.items():
                    if k.startswith(target_prefix):
                        new_key = k.replace(target_prefix, replacement_prefix, 1)
                        new_alphas[new_key] = v
                    else:
                        new_alphas[k] = v
                    network_alphas = new_alphas

        return original_load_lora_into_text_encoder(
            state_dict=state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder,
            prefix=prefix,
            lora_scale=lora_scale,
            text_encoder_name=text_encoder_name,
            adapter_name=adapter_name,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            metadata=metadata,
        )

    diffusers.loaders.lora_base._load_lora_into_text_encoder = patched_load_lora_into_text_encoder
    diffusers.loaders.lora_pipeline._load_lora_into_text_encoder = patched_load_lora_into_text_encoder
except Exception as e:
    warnings.warn(f"Failed to apply text encoder LoRA loading monkey patch: {e}", RuntimeWarning)


DEFAULT_MODEL_ID = "OzzyGT/RealVisXL_V4.0_inpainting"
FALLBACK_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_SAM_MODEL_ID = "facebook/sam-vit-base"
DEFAULT_PROMPT = (
    "photorealistic face of a synthetic non-famous person, natural skin texture, "
    "realistic eyes, matching head pose, matching lighting, high detail"
)
DEFAULT_NEGATIVE_PROMPT = (
    "celebrity, famous person, same identity, cartoon, anime, 3d render, doll, "
    "plastic skin, deformed face, asymmetrical eyes, bad anatomy, mask artifact, "
    "uncanny, blurry"
)
DEFAULT_PRIVACY_NEGATIVE_PROMPT = (
    "realistic human face, photorealistic human skin, human identity, "
    "recognizable person, celebrity likeness, real person, face swap, deepfake, "
    "accurate human facial features"
)
DEFAULT_FIRE_PROMPT = (
    "the entire head and masked face area is replaced by a roaring supernatural fireball, "
    "no face visible, no facial features, opaque cinematic realistic flames, blazing fire texture, "
    "swirling orange flames, glowing embers, overlapping fire wisps, dark smoke, "
    "faceless anonymization VFX, warm fire lighting reflecting on hair and shoulders, "
    "high-resolution visual effects"
)
DEFAULT_FIRE_NEGATIVE_PROMPT = (
    "human face, realistic human face, photorealistic human skin, skin texture, "
    "recognizable person, eyes, nose, mouth, lips, teeth, facial features, "
    "face swap, deepfake, celebrity likeness, still face, portrait face, text, watermark"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MAX_TORCH_SEED = 2**32 - 1
CONTROLNET_MODEL_IDS = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
}


@dataclass(frozen=True)
class FaceDetection:
    frame_index: int
    track_id: str | None
    bbox: tuple[float, float, float, float]
    confidence: float | None = None
    polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    landmarks: tuple[tuple[float, float], ...] = ()
    synthetic_id: str | None = None


@dataclass(frozen=True)
class SyntheticIdentity:
    identity_id: str
    prompt: str = ""
    negative_prompt: str = ""
    lora: str | None = None
    lora_weight: float | None = None
    seed_offset: int = 0


def parse_id_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def parse_bbox(raw_bbox: Any, bbox_format: str) -> tuple[float, float, float, float]:
    if isinstance(raw_bbox, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(raw_bbox):
            return (
                float(raw_bbox["x1"]),
                float(raw_bbox["y1"]),
                float(raw_bbox["x2"]),
                float(raw_bbox["y2"]),
            )
        if {"x", "y", "w", "h"}.issubset(raw_bbox):
            x = float(raw_bbox["x"])
            y = float(raw_bbox["y"])
            return x, y, x + float(raw_bbox["w"]), y + float(raw_bbox["h"])
        raise ValueError(f"Unsupported bbox object keys: {sorted(raw_bbox)}")

    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(f"bbox must be a 4-value list/tuple, got: {raw_bbox!r}")

    a, b, c, d = (float(value) for value in raw_bbox)
    if bbox_format == "xywh":
        return a, b, a + c, b + d
    return a, b, c, d


def parse_flat_points(values: list[Any]) -> list[tuple[float, float]]:
    if not values:
        return []
    step = 2 if len(values) % 2 == 0 else 3
    if len(values) % step != 0:
        return []
    points = []
    for index in range(0, len(values), step):
        points.append((float(values[index]), float(values[index + 1])))
    return points


def parse_points(raw_points: Any) -> list[tuple[float, float]]:
    if raw_points is None:
        return []
    if isinstance(raw_points, dict):
        if {"x", "y"}.issubset(raw_points):
            return [(float(raw_points["x"]), float(raw_points["y"]))]
        for key in ("points", "landmarks", "keypoints", "polygon"):
            if key in raw_points:
                return parse_points(raw_points[key])
        return []
    if not isinstance(raw_points, (list, tuple)):
        return []
    if not raw_points:
        return []
    if all(isinstance(value, (int, float)) for value in raw_points):
        return parse_flat_points(list(raw_points))

    points: list[tuple[float, float]] = []
    for point in raw_points:
        if isinstance(point, dict) and {"x", "y"}.issubset(point):
            points.append((float(point["x"]), float(point["y"])))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            if isinstance(point[0], (int, float)) and isinstance(point[1], (int, float)):
                points.append((float(point[0]), float(point[1])))
    return points


def parse_polygons(raw_polygons: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    if raw_polygons is None:
        return ()
    if isinstance(raw_polygons, dict):
        for key in ("polygons", "segments", "segmentation", "points", "polygon"):
            if key in raw_polygons:
                return parse_polygons(raw_polygons[key])
        return ()
    if not isinstance(raw_polygons, (list, tuple)) or not raw_polygons:
        return ()
    if all(isinstance(value, (int, float)) for value in raw_polygons):
        points = parse_flat_points(list(raw_polygons))
        return (tuple(points),) if len(points) >= 3 else ()
    if parse_points(raw_polygons):
        points = parse_points(raw_polygons)
        return (tuple(points),) if len(points) >= 3 else ()

    polygons = []
    for polygon in raw_polygons:
        points = parse_points(polygon)
        if len(points) >= 3:
            polygons.append(tuple(points))
    return tuple(polygons)


def detection_track_id(raw: dict[str, Any]) -> str | None:
    for key in ("track_id", "trackId", "id", "track"):
        if key in raw and raw[key] is not None:
            return str(raw[key])
    return None


def detection_confidence(raw: dict[str, Any]) -> float | None:
    for key in ("confidence", "conf", "score"):
        if key in raw and raw[key] is not None:
            return float(raw[key])
    return None


def detection_polygons(raw: dict[str, Any]) -> tuple[tuple[tuple[float, float], ...], ...]:
    for key in ("segmentation", "segments", "polygon", "polygons", "mask_polygon"):
        if key in raw:
            return parse_polygons(raw[key])
    return ()


def detection_landmarks(raw: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    for key in ("landmarks", "face_landmarks", "keypoints", "points"):
        if key in raw:
            return tuple(parse_points(raw[key]))
    return ()


def detection_synthetic_id(raw: dict[str, Any]) -> str | None:
    for key in ("synthetic_id", "identity_id", "anon_identity", "target_identity"):
        if key in raw and raw[key] is not None:
            return str(raw[key])
    return None


def detection_bbox(raw: dict[str, Any], bbox_format: str) -> tuple[float, float, float, float]:
    for key in ("bbox", "box", "xyxy", "bounds"):
        if key in raw:
            return parse_bbox(raw[key], bbox_format)
    if {"x1", "y1", "x2", "y2"}.issubset(raw):
        return parse_bbox(raw, "xyxy")
    if {"x", "y", "w", "h"}.issubset(raw):
        return parse_bbox(raw, "xywh")
    raise ValueError(f"Detection has no bbox-like field: {raw!r}")


def frame_index_from_record(record: dict[str, Any], default: int) -> int:
    for key in ("frame_index", "frame_idx", "frame", "image_id"):
        if key in record and record[key] is not None:
            return int(record[key])
    return default


def load_detections(path: Path, bbox_format: str) -> dict[int, list[FaceDetection]]:
    """Load several common YOLO/tracker JSON layouts into a frame-index map."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_frame: dict[int, list[FaceDetection]] = {}

    def add_detection(frame_index: int, raw: dict[str, Any]) -> None:
        detection = FaceDetection(
            frame_index=frame_index,
            track_id=detection_track_id(raw),
            bbox=detection_bbox(raw, raw.get("bbox_format", bbox_format)),
            confidence=detection_confidence(raw),
            polygons=detection_polygons(raw),
            landmarks=detection_landmarks(raw),
            synthetic_id=detection_synthetic_id(raw),
        )
        by_frame.setdefault(frame_index, []).append(detection)

    if isinstance(payload, dict) and "frames" in payload:
        frames = payload["frames"]
        if isinstance(frames, dict):
            iterable = (
                {"frame_index": int(frame_index), "faces": faces}
                for frame_index, faces in frames.items()
            )
        else:
            iterable = frames
        for default_index, frame in enumerate(iterable):
            if not isinstance(frame, dict):
                raise ValueError(f"Frame record must be an object, got: {frame!r}")
            frame_index = frame_index_from_record(frame, default_index)
            faces = (
                frame.get("faces")
                or frame.get("detections")
                or frame.get("objects")
                or frame.get("tracks")
                or []
            )
            for raw in faces:
                add_detection(frame_index, raw)
        return by_frame

    if isinstance(payload, dict):
        for frame_index, faces in payload.items():
            if not isinstance(faces, list):
                continue
            for raw in faces:
                add_detection(int(frame_index), raw)
        return by_frame

    if isinstance(payload, list):
        for default_index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError(f"Detection record must be an object, got: {record!r}")
            if any(key in record for key in ("faces", "detections", "objects", "tracks")):
                frame_index = frame_index_from_record(record, default_index)
                faces = (
                    record.get("faces")
                    or record.get("detections")
                    or record.get("objects")
                    or record.get("tracks")
                    or []
                )
                for raw in faces:
                    add_detection(frame_index, raw)
            else:
                add_detection(frame_index_from_record(record, default_index), record)
        return by_frame

    raise ValueError("Unsupported detections JSON layout.")


def default_identity_bank() -> list[SyntheticIdentity]:
    return [
        SyntheticIdentity(
            identity_id="synthetic_neutral_01",
            prompt="synthetic non-famous Korean adult face, balanced facial features",
            seed_offset=11001,
        ),
        SyntheticIdentity(
            identity_id="synthetic_male_01",
            prompt="synthetic non-famous Korean man face, natural realistic portrait",
            seed_offset=21001,
        ),
        SyntheticIdentity(
            identity_id="synthetic_female_01",
            prompt="synthetic non-famous Korean woman face, natural realistic portrait",
            seed_offset=31001,
        ),
    ]


def load_identity_bank(path: Path | None) -> list[SyntheticIdentity]:
    if path is None:
        return default_identity_bank()

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_identities = payload.get("identities", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_identities, list):
        raise ValueError("Identity bank must be a list or an object with an 'identities' list.")

    identities = []
    for index, raw in enumerate(raw_identities):
        if not isinstance(raw, dict):
            raise ValueError(f"Identity record must be an object, got: {raw!r}")
        identity_id = str(raw.get("id") or raw.get("identity_id") or f"synthetic_{index:02d}")
        identities.append(
            SyntheticIdentity(
                identity_id=identity_id,
                prompt=str(raw.get("prompt", "")),
                negative_prompt=str(raw.get("negative_prompt", "")),
                lora=str(raw["lora"]) if raw.get("lora") else None,
                lora_weight=float(raw["lora_weight"]) if raw.get("lora_weight") is not None else None,
                seed_offset=int(raw.get("seed_offset", stable_track_offset(identity_id, index))),
            )
        )
    return identities or default_identity_bank()


def identity_adapter_name(identity: SyntheticIdentity) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in identity.identity_id)
    return f"identity_{safe}"


def identity_lookup(identities: list[SyntheticIdentity]) -> dict[str, SyntheticIdentity]:
    return {identity.identity_id: identity for identity in identities}


def choose_identity(
    detection: FaceDetection | None,
    identities: list[SyntheticIdentity],
    face_index: int,
) -> SyntheticIdentity:
    if not identities:
        return default_identity_bank()[0]
    by_id = identity_lookup(identities)
    if detection and detection.synthetic_id and detection.synthetic_id in by_id:
        return by_id[detection.synthetic_id]
    key = detection.track_id if detection and detection.track_id is not None else str(face_index)
    offset = stable_track_offset(key, face_index)
    return identities[offset % len(identities)]


def should_anonymize(
    detection: FaceDetection,
    keep_track_ids: set[str],
    anonymize_track_ids: set[str],
    anonymize_untracked: bool,
) -> bool:
    track_id = detection.track_id
    if track_id is None:
        return anonymize_untracked and not anonymize_track_ids
    if track_id in keep_track_ids:
        return False
    if anonymize_track_ids:
        return track_id in anonymize_track_ids
    return True


def active_keep_track_ids(args: argparse.Namespace) -> set[str]:
    return getattr(args, "_active_keep_track_ids", args.keep_track_ids)


def selected_detections(
    detections: Iterable[FaceDetection],
    keep_track_ids: set[str],
    anonymize_track_ids: set[str],
    anonymize_untracked: bool,
) -> list[FaceDetection]:
    selected = [
        detection
        for detection in detections
        if should_anonymize(
            detection,
            keep_track_ids=keep_track_ids,
            anonymize_track_ids=anonymize_track_ids,
            anonymize_untracked=anonymize_untracked,
        )
    ]
    return sorted(selected, key=lambda detection: bbox_area(detection.bbox), reverse=True)


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clamp_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    expansion: float,
    y_shift: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5 + box_h * y_shift
    new_w = box_w * expansion
    new_h = box_h * expansion

    left = max(0, int(round(cx - new_w * 0.5)))
    top = max(0, int(round(cy - new_h * 0.5)))
    right = min(width, int(round(cx + new_w * 0.5)))
    bottom = min(height, int(round(cy + new_h * 0.5)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_box_for_face(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    expansion: float,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(box_w, box_h) * expansion
    side = max(side, float(min_size))
    side = min(side, float(max(width, height)))

    left = int(round(cx - side * 0.5))
    top = int(round(cy - side * 0.5))
    right = int(round(cx + side * 0.5))
    bottom = int(round(cy + side * 0.5))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def shift_detection_to_crop(
    detection: FaceDetection,
    crop_box: tuple[int, int, int, int],
) -> FaceDetection:
    left, top, _, _ = crop_box
    x1, y1, x2, y2 = detection.bbox
    polygons = tuple(
        tuple((x - left, y - top) for x, y in polygon)
        for polygon in detection.polygons
    )
    landmarks = tuple((x - left, y - top) for x, y in detection.landmarks)
    return FaceDetection(
        frame_index=detection.frame_index,
        track_id=detection.track_id,
        bbox=(x1 - left, y1 - top, x2 - left, y2 - top),
        confidence=detection.confidence,
        polygons=polygons,
        landmarks=landmarks,
        synthetic_id=detection.synthetic_id,
    )


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def scale_points_to_image(
    points: Iterable[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    raw_points = list(points)
    if not raw_points:
        return []
    max_coord = max(max(abs(x), abs(y)) for x, y in raw_points)
    is_normalized = max_coord <= 1.5
    scaled = []
    for x, y in raw_points:
        px = x * width if is_normalized else x
        py = y * height if is_normalized else y
        scaled.append(
            (
                int(round(min(max(px, 0.0), width - 1))),
                int(round(min(max(py, 0.0), height - 1))),
            )
        )
    return scaled


def draw_polygon_detection(
    draw: ImageDraw.ImageDraw,
    detection: FaceDetection,
    width: int,
    height: int,
) -> bool:
    drawn = False
    for polygon in detection.polygons:
        points = scale_points_to_image(polygon, width, height)
        if len(points) >= 3:
            draw.polygon(points, fill=255)
            drawn = True
    return drawn


def draw_landmark_detection(mask_array: np.ndarray, detection: FaceDetection) -> bool:
    if len(detection.landmarks) < 3:
        return False
    height, width = mask_array.shape[:2]
    points = scale_points_to_image(detection.landmarks, width, height)
    if len(points) < 3:
        return False
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    cv2.fillConvexPoly(mask_array, hull, 255)
    return True


def draw_ellipse_detection(
    draw: ImageDraw.ImageDraw,
    detection: FaceDetection,
    width: int,
    height: int,
    mask_expansion: float,
    mask_y_shift: float,
) -> bool:
    box = clamp_bbox(
        detection.bbox,
        width=width,
        height=height,
        expansion=mask_expansion,
        y_shift=mask_y_shift,
    )
    if box is None:
        return False
    draw.ellipse(box, fill=255)
    return True


def draw_bbox_detection(
    draw: ImageDraw.ImageDraw,
    detection: FaceDetection,
    width: int,
    height: int,
    mask_expansion: float,
    mask_y_shift: float,
) -> bool:
    box = clamp_bbox(
        detection.bbox,
        width=width,
        height=height,
        expansion=mask_expansion,
        y_shift=mask_y_shift,
    )
    if box is None:
        return False
    draw.rectangle(box, fill=255)
    return True


def load_sam_segmenter(args: argparse.Namespace) -> tuple[Any, Any, str]:
    if hasattr(args, "_sam_model") and hasattr(args, "_sam_processor") and hasattr(args, "_sam_device"):
        return args._sam_model, args._sam_processor, args._sam_device

    import torch
    from transformers import SamModel, SamProcessor

    device = choose_device(args.sam_device)
    dtype = torch_dtype_for_device(device)
    model = SamModel.from_pretrained(
        args.sam_model_id,
        torch_dtype=dtype,
        local_files_only=args.sam_local_files_only,
    )
    model.to(device)
    model.eval()
    processor = SamProcessor.from_pretrained(
        args.sam_model_id,
        local_files_only=args.sam_local_files_only,
    )
    args._sam_model = model
    args._sam_processor = processor
    args._sam_device = device
    return model, processor, device


def draw_sam_detection(
    binary_mask: Image.Image,
    source_image: Image.Image | None,
    detection: FaceDetection,
    mask_expansion: float,
    mask_y_shift: float,
    args: argparse.Namespace | None,
) -> bool:
    if source_image is None or args is None:
        return False

    box = clamp_bbox(
        detection.bbox,
        width=source_image.width,
        height=source_image.height,
        expansion=mask_expansion,
        y_shift=mask_y_shift,
    )
    if box is None:
        return False

    import torch

    model, processor, device = load_sam_segmenter(args)

    input_points = None
    input_labels = None

    if detection.landmarks and len(detection.landmarks) > 0:
        points = scale_points_to_image(detection.landmarks, source_image.width, source_image.height)
        input_points = [[[[float(x), float(y)] for x, y in points]]]
        input_labels = [[[1] * len(points)]]
    else:
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        approx_kps = [
            (x1 + width * 0.34, y1 + height * 0.38),
            (x1 + width * 0.66, y1 + height * 0.38),
            (x1 + width * 0.50, y1 + height * 0.55),
            (x1 + width * 0.38, y1 + height * 0.74),
            (x1 + width * 0.62, y1 + height * 0.74),
        ]
        input_points = [[[[float(x), float(y)] for x, y in approx_kps]]]
        input_labels = [[[1] * len(approx_kps)]]

    inputs = processor(
        source_image.convert("RGB"),
        input_boxes=[[[float(value) for value in box]]],
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )
    dtype = torch_dtype_for_device(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(dtype)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=args.sam_multimask)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
        mask_threshold=args.sam_mask_threshold,
    )[0]
    iou_scores = outputs.iou_scores.detach().cpu()[0, 0]
    mask_index = int(torch.argmax(iou_scores).item()) if args.sam_multimask else 0
    mask_array = masks[0, mask_index].numpy().astype(np.uint8) * 255
    if mask_array.max() == 0:
        return False

    current = np.array(binary_mask, dtype=np.uint8)
    binary_mask.paste(Image.fromarray(np.maximum(current, mask_array), mode="L"))
    return True


def draw_detection_mask(
    binary_mask: Image.Image,
    source_image: Image.Image | None,
    detection: FaceDetection,
    mask_mode: str,
    mask_fallback: str,
    mask_expansion: float,
    mask_y_shift: float,
    args: argparse.Namespace | None = None,
) -> bool:
    width, height = binary_mask.size
    draw = ImageDraw.Draw(binary_mask)

    if mask_mode == "sam":
        drawn = draw_sam_detection(
            binary_mask=binary_mask,
            source_image=source_image,
            detection=detection,
            mask_expansion=mask_expansion,
            mask_y_shift=mask_y_shift,
            args=args,
        )
        if drawn:
            return True

    if mask_mode == "bbox":
        return draw_bbox_detection(
            draw,
            detection,
            width=width,
            height=height,
            mask_expansion=mask_expansion,
            mask_y_shift=mask_y_shift,
        )

    if mask_mode in {"auto", "segmentation"} and detection.polygons:
        return draw_polygon_detection(draw, detection, width, height)

    if mask_mode in {"auto", "landmark"} and detection.landmarks:
        mask_array = np.array(binary_mask, dtype=np.uint8)
        drawn = draw_landmark_detection(mask_array, detection)
        if drawn:
            binary_mask.paste(Image.fromarray(mask_array, mode="L"))
            return True

    if mask_mode == "ellipse" or (mask_fallback == "ellipse" and mask_mode != "ellipse"):
        return draw_ellipse_detection(
            draw,
            detection,
            width=width,
            height=height,
            mask_expansion=mask_expansion,
            mask_y_shift=mask_y_shift,
        )
    return False


def build_face_mask(
    image: Image.Image | None,
    image_size: tuple[int, int],
    detections: Iterable[FaceDetection],
    keep_track_ids: set[str],
    anonymize_track_ids: set[str],
    anonymize_untracked: bool,
    mask_mode: str,
    mask_fallback: str,
    mask_expansion: float,
    mask_y_shift: float,
    mask_dilation: int,
    mask_blur: int,
    args: argparse.Namespace | None = None,
) -> Image.Image:
    """Build a soft inpainting mask from selected tracked face regions."""
    width, height = image_size
    binary_mask = Image.new("L", (width, height), 0)

    for detection in detections:
        if not should_anonymize(
            detection,
            keep_track_ids=keep_track_ids,
            anonymize_track_ids=anonymize_track_ids,
            anonymize_untracked=anonymize_untracked,
        ):
            continue
        draw_detection_mask(
            binary_mask=binary_mask,
            source_image=image,
            detection=detection,
            mask_mode=mask_mode,
            mask_fallback=mask_fallback,
            mask_expansion=mask_expansion,
            mask_y_shift=mask_y_shift,
            args=args,
        )

    mask_array = np.array(binary_mask, dtype=np.uint8)
    if mask_dilation > 0 and mask_array.max() > 0:
        kernel_size = odd_kernel(mask_dilation)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)

    mask = Image.fromarray(mask_array, mode="L")
    if mask_blur > 0 and mask_array.max() > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    return mask


def resize_for_model(
    image: Image.Image,
    mask: Image.Image,
    max_side: int,
) -> tuple[Image.Image, Image.Image]:
    if max_side <= 0:
        return image, mask
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        model_width = width - (width % 8)
        model_height = height - (height % 8)
    else:
        scale = max_side / float(long_side)
        model_width = int(math.floor(width * scale / 8.0) * 8)
        model_height = int(math.floor(height * scale / 8.0) * 8)

    model_width = max(64, model_width)
    model_height = max(64, model_height)
    if (model_width, model_height) == image.size:
        return image, mask
    return (
        image.resize((model_width, model_height), Image.Resampling.LANCZOS),
        mask.resize((model_width, model_height), Image.Resampling.LANCZOS),
    )


def fire_prepaint_image(
    image: Image.Image,
    mask: Image.Image,
    args: argparse.Namespace,
    detection: FaceDetection | None = None,
) -> Image.Image:
    if args.replacement_preset != "fire" or args.fire_prepaint == "none":
        return image
    fill_colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
    }
    fill = Image.new("RGB", image.size, fill_colors[args.fire_prepaint])
    prepaint_mask = mask
    if args.fire_prepaint_region == "bbox" and detection is not None:
        box = clamp_bbox(
            detection.bbox,
            width=image.width,
            height=image.height,
            expansion=args.fire_prepaint_bbox_expansion,
            y_shift=0.0,
        )
        if box is not None:
            prepaint_mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(prepaint_mask).rectangle(box, fill=255)
    return Image.composite(fill, image.convert("RGB"), prepaint_mask)


def composite_inpaint(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    if mask_array.ndim == 2:
        mask_array = mask_array[..., None]
    original_array = np.asarray(original.convert("RGB"), dtype=np.float32)
    generated_array = np.asarray(generated.convert("RGB"), dtype=np.float32)
    composite = original_array * (1.0 - mask_array) + generated_array * mask_array
    return Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), mode="RGB")


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None] * 0.55
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    preview = base * (1.0 - alpha) + red * alpha
    return Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8), mode="RGB")


def mask_area_ratio(mask: Image.Image) -> float:
    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    if mask_array.size == 0:
        return 0.0
    return float((mask_array > 8).sum()) / float(mask_array.size)


def masked_mean_abs_delta(before: Image.Image, after: Image.Image, mask: Image.Image) -> float:
    before_array = np.asarray(before.convert("RGB"), dtype=np.float32)
    after_array = np.asarray(after.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8) > 8
    if before_array.shape != after_array.shape:
        after_array = np.asarray(after.resize(before.size).convert("RGB"), dtype=np.float32)
    if not mask_array.any():
        return 0.0
    delta = np.abs(after_array - before_array).mean(axis=2)
    return float(delta[mask_array].mean())


def fallback_anonymize(image: Image.Image, mask: Image.Image, args: argparse.Namespace) -> Image.Image:
    if args.fallback_mode == "none":
        return image
    if args.fallback_mode == "pixelate":
        small_width = max(1, image.width // args.fallback_pixel_size)
        small_height = max(1, image.height // args.fallback_pixel_size)
        anonymized = image.resize((small_width, small_height), Image.Resampling.BILINEAR)
        anonymized = anonymized.resize(image.size, Image.Resampling.NEAREST)
    else:
        anonymized = image.filter(ImageFilter.GaussianBlur(radius=args.fallback_blur_radius))
    return composite_inpaint(image, anonymized, mask)


def append_report(args: argparse.Namespace, record: dict[str, Any]) -> None:
    if not hasattr(args, "_report_records"):
        args._report_records = []
    args._report_records.append(record)


def record_quality(
    args: argparse.Namespace,
    frame_index: int,
    detection: FaceDetection | None,
    face_index: int,
    identity: SyntheticIdentity,
    mask: Image.Image,
    before: Image.Image,
    after: Image.Image,
    crop_box: tuple[int, int, int, int] | None,
    fallback_used: bool,
    fallback_reason: str | None,
) -> None:
    append_report(
        args,
        {
            "frame_index": frame_index,
            "track_id": detection.track_id if detection else None,
            "face_index": face_index,
            "identity_id": identity.identity_id,
            "crop_box": list(crop_box) if crop_box else None,
            "mask_area_ratio": round(mask_area_ratio(mask), 6),
            "mean_abs_delta": round(masked_mean_abs_delta(before, after, mask), 4),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        },
    )


def inpaint_or_fallback(
    pipe: Any,
    device: str,
    image: Image.Image,
    mask: Image.Image,
    seed: int | None,
    identity: SyntheticIdentity,
    frame_index: int,
    detection: FaceDetection | None,
    face_index: int,
    crop_box: tuple[int, int, int, int] | None,
    args: argparse.Namespace,
) -> Image.Image:
    area_ratio = mask_area_ratio(mask)
    fallback_used = False
    fallback_reason = None
    if area_ratio < args.min_mask_area_ratio:
        result = fallback_anonymize(image, mask, args)
        fallback_used = args.fallback_mode != "none"
        fallback_reason = "mask_too_small"
    else:
        try:
            result = run_inpaint(pipe, device, image, mask, seed, identity, detection, args)
        except Exception as exc:
            if args.fallback_mode == "none":
                raise
            result = fallback_anonymize(image, mask, args)
            fallback_used = True
            fallback_reason = f"inpaint_error:{type(exc).__name__}"

    if fallback_reason is None:
        mean_delta = masked_mean_abs_delta(image, result, mask)
        if mean_delta < args.min_mean_delta and args.fallback_mode != "none":
            result = fallback_anonymize(image, mask, args)
            fallback_used = True
            fallback_reason = "low_mask_delta"

    record_quality(
        args=args,
        frame_index=frame_index,
        detection=detection,
        face_index=face_index,
        identity=identity,
        mask=mask,
        before=image,
        after=result,
        crop_box=crop_box,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    return result


def write_quality_report(args: argparse.Namespace) -> None:
    if not args.report_json:
        return
    records = getattr(args, "_report_records", [])
    fallback_count = sum(1 for record in records if record.get("fallback_used"))
    payload = {
        "input": str(args.input),
        "output": str(args.output) if args.output else None,
        "model_id": args.model_id,
        "replacement_preset": args.replacement_preset,
        "controlnet": args.controlnet,
        "inpaint_scope": args.inpaint_scope,
        "mask_mode": args.mask_mode,
        "record_count": len(records),
        "fallback_count": fallback_count,
        "records": records,
    }
    if args.mask_mode == "sam":
        payload["sam"] = {
            "model_id": args.sam_model_id,
            "device": getattr(args, "_sam_device", args.sam_device),
            "mask_threshold": args.sam_mask_threshold,
            "multimask": args.sam_multimask,
            "local_files_only": args.sam_local_files_only,
        }
    owner_profile = getattr(args, "_owner_profile", None)
    owner_frames = getattr(args, "_owner_frame_records", [])
    if owner_profile is not None:
        payload["owner"] = owner_profile.to_dict(
            include_embedding=args.include_owner_embedding_in_report,
        )
        payload["owner_matching"] = {
            "high_threshold": args.owner_high_threshold,
            "low_threshold": args.owner_low_threshold,
            "vote_window": args.owner_vote_window,
            "min_votes": args.owner_min_votes,
            "hold_frames": args.owner_hold_frames,
            "frames": owner_frames,
        }
    reference_prompt_payload = getattr(args, "_reference_prompt_payload", None)
    if reference_prompt_payload is not None:
        payload["reference_prompt"] = {
            "reference_count": reference_prompt_payload.get("reference_count"),
            "mode": reference_prompt_payload.get("mode"),
            "prompt": reference_prompt_payload.get("prompt"),
            "shared_tags": reference_prompt_payload.get("shared_tags", []),
            "captions": reference_prompt_payload.get("captions", []),
            "warnings": reference_prompt_payload.get("warnings", []),
        }
    reference_face_bank = getattr(args, "_reference_face_bank", None)
    if reference_face_bank is not None:
        payload["reference_face_conditioning"] = {
            "reference_count": len(reference_face_bank.faces),
            "crop_mode": reference_face_bank.crop_mode,
            "target_expansion": reference_face_bank.target_expansion,
            "feather": reference_face_bank.feather,
            "opacity": reference_face_bank.opacity,
        }
    reference_identity = getattr(args, "_reference_identity_condition", None)
    if reference_identity is not None:
        payload["reference_identity_route"] = {
            "route": reference_identity.route,
            "reference_count": len(reference_identity.image_paths),
            "detected_human_faces": len(reference_identity.face_analyses),
            "human_scores": [round(face.score, 4) for face in reference_identity.face_analyses],
            "instantid_controlnet_scale": args.instantid_controlnet_scale,
            "instantid_ip_adapter_scale": args.instantid_ip_adapter_scale,
            "ip_adapter_scale": args.ip_adapter_scale,
        }
    if args.replacement_preset == "fire":
        payload["fire"] = {
            "prepaint": args.fire_prepaint,
            "prepaint_region": args.fire_prepaint_region,
            "prepaint_bbox_expansion": args.fire_prepaint_bbox_expansion,
            "force_bbox": args.fire_force_bbox,
        }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype_for_device(device: str) -> Any:
    import torch

    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def controlnet_model_id(args: argparse.Namespace) -> str:
    if args.controlnet_model_id:
        return args.controlnet_model_id
    return CONTROLNET_MODEL_IDS[args.controlnet]


def apply_scheduler(pipe: Any, scheduler_name: str) -> None:
    if scheduler_name == "default":
        return
    if scheduler_name == "dpmpp_2m_karras":
        from diffusers import DPMSolverMultistepScheduler

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        return
    if scheduler_name == "euler_a":
        from diffusers import EulerAncestralDiscreteScheduler

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def is_local_path(path: str | None) -> bool:
    return bool(path) and Path(path).expanduser().exists()


def instantid_adapter_path(args: argparse.Namespace) -> str:
    if args.instantid_adapter_path:
        return str(args.instantid_adapter_path)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("InstantID adapter auto-download requires huggingface_hub.") from exc
    return hf_hub_download(
        repo_id=args.instantid_repo_id,
        filename=args.instantid_adapter_filename,
    )


def load_instantid_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers.models import ControlNetModel

    if args.instantid_pipeline_dir:
        sys.path.insert(0, str(args.instantid_pipeline_dir))
    try:
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
    except ImportError as exc:
        raise RuntimeError(
            "InstantID route requires the official InstantID "
            "pipeline_stable_diffusion_xl_instantid.py on PYTHONPATH or --instantid-pipeline-dir."
        ) from exc

    device = choose_device(args.device)
    dtype = torch_dtype_for_device(device)
    controlnet_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
    }
    controlnet_path = str(args.instantid_controlnet_model)
    if args.instantid_controlnet_subfolder and not is_local_path(controlnet_path):
        controlnet_kwargs["subfolder"] = args.instantid_controlnet_subfolder
    elif args.instantid_controlnet_subfolder and (
        Path(controlnet_path) / args.instantid_controlnet_subfolder
    ).exists():
        controlnet_path = str(Path(controlnet_path) / args.instantid_controlnet_subfolder)

    controlnet = ControlNetModel.from_pretrained(controlnet_path, **controlnet_kwargs)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        args.instantid_base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.load_ip_adapter_instantid(instantid_adapter_path(args))
    apply_scheduler(pipe, args.scheduler)

    if device == "cuda" and args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if args.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    return pipe, device


def configure_ip_adapter(pipe: Any, args: argparse.Namespace) -> None:
    condition = getattr(args, "_reference_identity_condition", None)
    if condition is None or not condition.is_ip_adapter:
        return
    if not hasattr(pipe, "load_ip_adapter"):
        raise RuntimeError("The active Diffusers pipeline does not support load_ip_adapter().")
    pipe.load_ip_adapter(
        args.ip_adapter_model,
        subfolder=args.ip_adapter_subfolder,
        weight_name=args.ip_adapter_weight_name,
    )
    if hasattr(pipe, "set_ip_adapter_scale"):
        pipe.set_ip_adapter_scale(args.ip_adapter_scale)


def ip_adapter_needs_image_encoder(args: argparse.Namespace) -> bool:
    weight_name = str(args.ip_adapter_weight_name or "")
    return "vit-h" in weight_name or "plus" in weight_name


def load_ip_adapter_image_encoder(args: argparse.Namespace, dtype: Any) -> Any | None:
    condition = getattr(args, "_reference_identity_condition", None)
    if condition is None or not condition.is_ip_adapter:
        return None
    if not args.ip_adapter_image_encoder_model and not ip_adapter_needs_image_encoder(args):
        return None
    from transformers import CLIPVisionModelWithProjection

    model_id = args.ip_adapter_image_encoder_model or args.ip_adapter_model
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.ip_adapter_image_encoder_subfolder:
        kwargs["subfolder"] = args.ip_adapter_image_encoder_subfolder
    return CLIPVisionModelWithProjection.from_pretrained(model_id, **kwargs)


def load_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import AutoPipelineForInpainting

    device = choose_device(args.device)
    dtype = torch_dtype_for_device(device)
    reference_condition = getattr(args, "_reference_identity_condition", None)
    if reference_condition is not None and reference_condition.is_instantid:
        return load_instantid_pipeline(args)

    pipeline_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if getattr(args, "variant", None) is not None:
        pipeline_kwargs["variant"] = args.variant
    elif dtype == torch.float16:
        pipeline_kwargs["variant"] = "fp16"
    image_encoder = load_ip_adapter_image_encoder(args, dtype)
    if image_encoder is not None:
        pipeline_kwargs["image_encoder"] = image_encoder

    if args.controlnet == "none":
        pipe = AutoPipelineForInpainting.from_pretrained(
            args.model_id,
            **pipeline_kwargs,
        )
    else:
        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id(args),
                torch_dtype=dtype,
                use_safetensors=True,
            )
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                args.model_id,
                controlnet=controlnet,
                **pipeline_kwargs,
            )
        except Exception:
            if not args.controlnet_fallback:
                raise
            warnings.warn("ControlNet load failed; falling back to plain SDXL inpainting.", RuntimeWarning)
            args.controlnet = "none"
            pipe = AutoPipelineForInpainting.from_pretrained(
                args.model_id,
                **pipeline_kwargs,
            )
    apply_scheduler(pipe, args.scheduler)
    if args.lora:
        pipe.load_lora_weights(args.lora, adapter_name=args.lora_adapter_name)
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([args.lora_adapter_name], adapter_weights=[args.lora_weight])
    loaded_adapters = {args.lora_adapter_name} if args.lora else set()
    for identity in args.identity_bank_entries:
        adapter_name = identity_adapter_name(identity)
        if not identity.lora or adapter_name in loaded_adapters:
            continue
        pipe.load_lora_weights(identity.lora, adapter_name=adapter_name)
        loaded_adapters.add(adapter_name)

    configure_ip_adapter(pipe, args)

    if device == "cuda" and args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    ip_adapter_active = reference_condition is not None and reference_condition.is_ip_adapter
    if args.attention_slicing and not ip_adapter_active and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe, device


def apply_identity_adapter(pipe: Any, args: argparse.Namespace, identity: SyntheticIdentity) -> None:
    if identity.lora and hasattr(pipe, "set_adapters"):
        pipe.set_adapters(
            [identity_adapter_name(identity)],
            adapter_weights=[identity.lora_weight if identity.lora_weight is not None else args.lora_weight],
        )
    elif args.lora and hasattr(pipe, "set_adapters"):
        pipe.set_adapters([args.lora_adapter_name], adapter_weights=[args.lora_weight])


def prompt_for_identity(args: argparse.Namespace, identity: SyntheticIdentity) -> tuple[str, str]:
    reference_prompt = getattr(args, "_reference_prompt_text", "")
    reference_condition = getattr(args, "_reference_identity_condition", None)

    if args.replacement_preset == "fire":
        user_prompt = args.prompt if args.prompt != DEFAULT_PROMPT else ""
        prompt_parts = [part for part in (args.fire_prompt, reference_prompt, user_prompt) if part]
        negative_parts = [
            part
            for part in (
                DEFAULT_PRIVACY_NEGATIVE_PROMPT,
                args.fire_negative_prompt,
                args.negative_prompt,
                identity.negative_prompt,
            )
            if part
        ]
        return ", ".join(prompt_parts), ", ".join(negative_parts)

    is_character_route = reference_condition is not None and reference_condition.is_ip_adapter

    if is_character_route:
        # For character/animal replacement, exclude human face prompts
        character_prompt = args.reference_character_prompt
        
        # Check if the character prompt or auto-caption contains "panda"
        is_panda = "panda" in character_prompt.lower()
        
        # 1. Expand the character prompt to avoid raccoon-like generations if it's a panda
        if is_panda:
            panda_keywords = "black and white giant panda face, black and white fur, white snout, deep black eye patches"
            if "giant panda" not in character_prompt.lower():
                character_prompt = f"{character_prompt}, {panda_keywords}"
                
        # 2. Append pose, lighting, and quality keywords for character routing to align with target image
        quality_prompts = "matching head pose, matching lighting, photorealistic, high detail"
        character_prompt = f"{character_prompt}, {quality_prompts}"
        
        # If user explicitly passed a custom prompt, keep it; otherwise ignore the default human prompt
        user_prompt = args.prompt if args.prompt != DEFAULT_PROMPT else ""
        prompt_parts = [part for part in (character_prompt, reference_prompt, user_prompt) if part]
        
        # 3. Enhance negative prompt for character routing
        char_neg_parts = []
        if args.negative_prompt:
            char_neg_parts.append(args.negative_prompt)
            
        # Always exclude human skin/face characteristics to avoid blending with the man's peach skin
        char_neg_parts.append("human, human face, human skin, skin texture, peach skin")
        char_neg_parts.append(DEFAULT_PRIVACY_NEGATIVE_PROMPT)
        
        # Exclude raccoon/badger features if it's a panda
        if is_panda:
            char_neg_parts.append("raccoon, red panda, brown fur, grey fur, orange fur, badger, brown face, yellow eyes, fox, dog, monkey, cat")
            
        negative_parts = [part for part in char_neg_parts if part]
    else:
        prompt_parts = [part for part in (identity.prompt, reference_prompt, args.prompt) if part]
        negative_parts = [part for part in (args.negative_prompt, identity.negative_prompt) if part]
        
    return ", ".join(prompt_parts), ", ".join(negative_parts)


def stable_track_offset(track_id: str | None, face_index: int) -> int:
    if track_id is None:
        return (face_index + 1) * 1009
    digest = hashlib.blake2b(str(track_id).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def seed_for_context(
    base_seed: int | None,
    frame_index: int,
    track_id: str | None,
    face_index: int,
    identity: SyntheticIdentity | None,
    seed_strategy: str,
    vary_seed_per_frame: bool,
) -> int | None:
    if base_seed is None:
        return None

    strategy = "frame" if vary_seed_per_frame and seed_strategy == "global" else seed_strategy
    seed = int(base_seed)
    if identity is not None:
        seed += identity.seed_offset
    if strategy in {"track", "frame-track"}:
        seed += stable_track_offset(track_id, face_index)
    if strategy in {"frame", "frame-track"}:
        seed += frame_index * 1_000_003
    return seed % MAX_TORCH_SEED


def generator_for_seed(seed: int | None, device: str) -> Any:
    if seed is None:
        return None
    import torch

    generator_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def make_canny_control_image(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    image_array = np.asarray(image.convert("RGB"))
    edges = cv2.Canny(image_array, args.canny_low, args.canny_high)
    edges = np.repeat(edges[:, :, None], 3, axis=2)
    return Image.fromarray(edges, mode="RGB")


def load_depth_models(args: argparse.Namespace, device: str) -> tuple[Any, Any]:
    if hasattr(args, "_depth_estimator") and hasattr(args, "_depth_processor"):
        return args._depth_estimator, args._depth_processor

    import torch
    from transformers import DPTForDepthEstimation

    try:
        from transformers import DPTImageProcessor
    except ImportError:
        from transformers import DPTFeatureExtractor as DPTImageProcessor

    processor = DPTImageProcessor.from_pretrained(args.depth_estimator_model)
    estimator = DPTForDepthEstimation.from_pretrained(args.depth_estimator_model).to(device)
    estimator.eval()
    args._depth_processor = processor
    args._depth_estimator = estimator
    return estimator, processor


def make_depth_control_image(image: Image.Image, args: argparse.Namespace, device: str) -> Image.Image:
    import torch
    import torch.nn.functional as F

    estimator, processor = load_depth_models(args, device)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast("cuda"):
                depth = estimator(pixel_values).predicted_depth
        else:
            depth = estimator(pixel_values).predicted_depth
    depth = F.interpolate(
        depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth, dim=[1, 2, 3], keepdim=True)
    depth = (depth - depth_min) / torch.clamp(depth_max - depth_min, min=1e-6)
    depth = torch.cat([depth] * 3, dim=1)
    depth = depth.permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray((depth * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def make_control_image(image: Image.Image, args: argparse.Namespace, device: str) -> Image.Image | None:
    if args.controlnet == "none":
        return None
    if args.controlnet == "canny":
        return make_canny_control_image(image, args)
    if args.controlnet == "depth":
        return make_depth_control_image(image, args, device)
    raise ValueError(f"Unsupported controlnet mode: {args.controlnet}")


def supports_call_argument(pipe: Any, argument_name: str) -> bool:
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return True
    return argument_name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def instantid_prompt_for_args(args: argparse.Namespace) -> tuple[str, str]:
    return args.instantid_prompt, args.instantid_negative_prompt


def run_instantid_generate(
    pipe: Any,
    device: str,
    image: Image.Image,
    mask: Image.Image,
    seed: int | None,
    detection: FaceDetection | None,
    args: argparse.Namespace,
) -> Image.Image:
    condition: ReferenceIdentityCondition | None = getattr(args, "_reference_identity_condition", None)
    if condition is None or condition.face_embedding is None or condition.analyzer is None:
        raise RuntimeError("InstantID route is active but reference identity conditioning is missing.")
    if detection is None:
        keypoints = condition.analyzer.target_keypoints(
            image,
            FaceDetection(frame_index=0, track_id=None, bbox=(0, 0, image.width, image.height)),
            min_confidence=args.reference_target_face_threshold,
        )
    else:
        keypoints = condition.analyzer.target_keypoints(
            image,
            detection,
            min_confidence=args.reference_target_face_threshold,
        )

    model_image, model_mask = resize_for_model(image, mask, max_side=args.max_side)
    scale_x = model_image.width / float(image.width)
    scale_y = model_image.height / float(image.height)
    model_keypoints = keypoints.copy()
    model_keypoints[:, 0] *= scale_x
    model_keypoints[:, 1] *= scale_y
    condition_image = draw_landmark_condition(model_image.size, model_keypoints)
    generator = generator_for_seed(seed=seed, device=device)
    prompt, negative_prompt = instantid_prompt_for_args(args)
    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_embeds": condition.face_embedding,
        "image": condition_image,
        "init_image": model_image,
        "mask_image": model_mask,
        "controlnet_conditioning_scale": args.instantid_controlnet_scale,
        "ip_adapter_scale": args.instantid_ip_adapter_scale,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "height": model_image.height,
        "width": model_image.width,
    }
    call_kwargs = {
        key: value
        for key, value in call_kwargs.items()
        if key in {"prompt", "negative_prompt"} or supports_call_argument(pipe, key)
    }
    generated = pipe(**call_kwargs).images[0]
    if generated.size != image.size:
        generated = generated.resize(image.size, Image.Resampling.LANCZOS)
    return composite_inpaint(image, generated, mask)


def run_inpaint(
    pipe: Any,
    device: str,
    image: Image.Image,
    mask: Image.Image,
    seed: int | None,
    identity: SyntheticIdentity,
    detection: FaceDetection | None,
    args: argparse.Namespace,
) -> Image.Image:
    if np.asarray(mask).max() == 0:
        return image.copy()

    reference_condition = getattr(args, "_reference_identity_condition", None)
    if reference_condition is not None and reference_condition.is_instantid:
        return run_instantid_generate(
            pipe=pipe,
            device=device,
            image=image,
            mask=mask,
            seed=seed,
            detection=detection,
            args=args,
        )

    apply_identity_adapter(pipe, args, identity)
    prompt, negative_prompt = prompt_for_identity(args, identity)
    print(f"DEBUG PROMPT: {prompt}")
    print(f"DEBUG NEGATIVE PROMPT: {negative_prompt}")
    inpaint_source = fire_prepaint_image(image, mask, args, detection=detection)
    model_image, model_mask = resize_for_model(inpaint_source, mask, max_side=args.max_side)
    generator = generator_for_seed(seed=seed, device=device)
    call_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": model_image,
        "mask_image": model_mask,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": generator,
    }
    control_image = make_control_image(model_image, args, device)
    if control_image is not None:
        call_kwargs["control_image"] = control_image
        call_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
        call_kwargs["guess_mode"] = args.controlnet_guess_mode
    if reference_condition is not None and reference_condition.is_ip_adapter:
        if reference_condition.ip_adapter_image is None:
            raise RuntimeError("IP-Adapter route is active but no reference image was prepared.")
        call_kwargs["ip_adapter_image"] = reference_condition.ip_adapter_image
    result = pipe(
        **call_kwargs,
    ).images[0]
    return composite_inpaint(image, result, mask)


def run_full_frame_inpaint(
    pipe: Any,
    device: str,
    image: Image.Image,
    detections: Iterable[FaceDetection],
    frame_index: int,
    args: argparse.Namespace,
) -> Image.Image:
    mask = build_face_mask(
        image=image,
        image_size=image.size,
        detections=detections,
        keep_track_ids=active_keep_track_ids(args),
        anonymize_track_ids=args.anonymize_track_ids,
        anonymize_untracked=args.anonymize_untracked,
        mask_mode=args.mask_mode,
        mask_fallback=args.mask_fallback,
        mask_expansion=args.mask_expansion,
        mask_y_shift=args.mask_y_shift,
        mask_dilation=args.mask_dilation,
        mask_blur=args.mask_blur,
        args=args,
    )
    identity = choose_identity(None, args.identity_bank_entries, 0)
    seed = seed_for_context(
        base_seed=args.seed,
        frame_index=frame_index,
        track_id=None,
        face_index=0,
        identity=identity,
        seed_strategy=args.seed_strategy,
        vary_seed_per_frame=args.vary_seed_per_frame,
    )
    return inpaint_or_fallback(
        pipe=pipe,
        device=device,
        image=image,
        mask=mask,
        seed=seed,
        identity=identity,
        frame_index=frame_index,
        detection=None,
        face_index=0,
        crop_box=None,
        args=args,
    )


def run_face_crop_inpaint(
    pipe: Any,
    device: str,
    image: Image.Image,
    detections: Iterable[FaceDetection],
    frame_index: int,
    args: argparse.Namespace,
) -> Image.Image:
    output = image.copy()
    targets = selected_detections(
        detections,
        keep_track_ids=active_keep_track_ids(args),
        anonymize_track_ids=args.anonymize_track_ids,
        anonymize_untracked=args.anonymize_untracked,
    )

    for face_index, detection in enumerate(targets):
        crop_box = crop_box_for_face(
            detection.bbox,
            width=output.width,
            height=output.height,
            expansion=args.crop_expansion,
            min_size=args.crop_min_size,
        )
        if crop_box is None:
            continue

        crop = output.crop(crop_box)
        local_detection = shift_detection_to_crop(detection, crop_box)
        reference_face_bank = getattr(args, "_reference_face_bank", None)
        if reference_face_bank is not None:
            crop = reference_face_bank.apply_to_crop(
                crop=crop,
                detection=local_detection,
                track_id=detection.track_id,
                face_index=face_index,
            )
        local_mask = build_face_mask(
            image=crop,
            image_size=crop.size,
            detections=[local_detection],
            keep_track_ids=set(),
            anonymize_track_ids=set(),
            anonymize_untracked=True,
            mask_mode=args.mask_mode,
            mask_fallback=args.mask_fallback,
            mask_expansion=args.mask_expansion,
            mask_y_shift=args.mask_y_shift,
            mask_dilation=args.mask_dilation,
            mask_blur=args.mask_blur,
            args=args,
        )
        identity = choose_identity(detection, args.identity_bank_entries, face_index)
        seed = seed_for_context(
            base_seed=args.seed,
            frame_index=frame_index,
            track_id=detection.track_id,
            face_index=face_index,
            identity=identity,
            seed_strategy=args.seed_strategy,
            vary_seed_per_frame=args.vary_seed_per_frame,
        )
        inpainted_crop = inpaint_or_fallback(
            pipe=pipe,
            device=device,
            image=crop,
            mask=local_mask,
            seed=seed,
            identity=identity,
            frame_index=frame_index,
            detection=local_detection,
            face_index=face_index,
            crop_box=crop_box,
            args=args,
        )
        output.paste(inpainted_crop, crop_box)

    return output


def detections_for_frame(
    detections_by_frame: dict[int, list[FaceDetection]],
    frame_index: int,
    hold_last_detections: bool,
) -> list[FaceDetection]:
    if frame_index in detections_by_frame:
        return detections_by_frame[frame_index]
    if not hold_last_detections:
        return []
    available = [idx for idx in detections_by_frame if idx <= frame_index]
    if not available:
        return []
    return detections_by_frame[max(available)]


def process_image(args: argparse.Namespace, detections_by_frame: dict[int, list[FaceDetection]]) -> None:
    image = Image.open(args.input).convert("RGB")
    detections = detections_for_frame(
        detections_by_frame,
        frame_index=args.frame_index,
        hold_last_detections=args.hold_last_detections,
    )
    keep_track_ids = update_owner_keep_ids(args, image, detections, args.frame_index)
    mask = build_face_mask(
        image=image,
        image_size=image.size,
        detections=detections,
        keep_track_ids=keep_track_ids,
        anonymize_track_ids=args.anonymize_track_ids,
        anonymize_untracked=args.anonymize_untracked,
        mask_mode=args.mask_mode,
        mask_fallback=args.mask_fallback,
        mask_expansion=args.mask_expansion,
        mask_y_shift=args.mask_y_shift,
        mask_dilation=args.mask_dilation,
        mask_blur=args.mask_blur,
        args=args,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.save_mask)

    if args.mask_preview:
        overlay_mask(image, mask).save(args.output)
        return

    pipe, device = load_pipeline(args)
    if args.inpaint_scope == "face-crop":
        output = run_face_crop_inpaint(pipe, device, image, detections, args.frame_index, args)
    else:
        identity = choose_identity(None, args.identity_bank_entries, 0)
        seed = seed_for_context(
            base_seed=args.seed,
            frame_index=args.frame_index,
            track_id=None,
            face_index=0,
            identity=identity,
            seed_strategy=args.seed_strategy,
            vary_seed_per_frame=args.vary_seed_per_frame,
        )
        output = inpaint_or_fallback(
            pipe=pipe,
            device=device,
            image=image,
            mask=mask,
            seed=seed,
            identity=identity,
            frame_index=args.frame_index,
            detection=None,
            face_index=0,
            crop_box=None,
            args=args,
        )
    output.save(args.output)


def pil_from_bgr(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode="RGB")


def bgr_from_pil(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def parse_csv_tuple(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_input_frame(path: Path, kind: str, frame_index: int) -> Image.Image:
    if kind == "image":
        return Image.open(path).convert("RGB")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from video: {path}")
        return pil_from_bgr(frame)
    finally:
        capture.release()


def owner_source_detections(
    args: argparse.Namespace,
    detections_by_frame: dict[int, list[FaceDetection]],
) -> list[FaceDetection]:
    return detections_for_frame(
        detections_by_frame,
        frame_index=args.owner_source_frame,
        hold_last_detections=args.hold_last_detections,
    )


def maybe_export_owner_crops(
    args: argparse.Namespace,
    detections_by_frame: dict[int, list[FaceDetection]],
    kind: str,
) -> None:
    if not args.owner_crops_dir:
        return
    image = load_input_frame(args.input, kind=kind, frame_index=args.owner_source_frame)
    detections = owner_source_detections(args, detections_by_frame)
    manifest_path = export_face_crops(
        image=image,
        detections=detections,
        output_dir=args.owner_crops_dir,
        source_frame=args.owner_source_frame,
        crop_expansion=args.owner_crop_expansion,
        crop_min_size=args.owner_crop_min_size,
    )
    args._owner_crops_manifest = manifest_path


def make_owner_embedder(args: argparse.Namespace) -> InsightFaceEmbedder:
    providers = args.face_recognition_providers or ("CPUExecutionProvider",)
    return InsightFaceEmbedder(
        model_name=args.face_recognition_model,
        providers=providers,
        det_size=args.face_recognition_det_size,
    )


def prepare_owner_matcher(
    args: argparse.Namespace,
    detections_by_frame: dict[int, list[FaceDetection]],
    kind: str,
) -> None:
    args._owner_matcher = None
    if args.owner_face_index is None and not args.owner_profile:
        return

    embedder = make_owner_embedder(args)
    if args.owner_face_index is not None:
        image = load_input_frame(args.input, kind=kind, frame_index=args.owner_source_frame)
        detections = owner_source_detections(args, detections_by_frame)
        profile = build_owner_profile(
            image=image,
            detections=detections,
            face_index=args.owner_face_index,
            embedder=embedder,
            source_frame=args.owner_source_frame,
            crop_expansion=args.owner_crop_expansion,
            crop_min_size=args.owner_crop_min_size,
        )
        if args.owner_profile:
            profile.write_json(args.owner_profile)
    else:
        profile = OwnerProfile.from_json(args.owner_profile)

    if profile.embedding_model != embedder.profile_model_name:
        warnings.warn(
            "Owner profile embedding model differs from the active face recognition model. "
            "Similarity scores may be invalid.",
            RuntimeWarning,
        )

    args._owner_matcher = OwnerMatcher(
        profile=profile,
        embedder=embedder,
        high_threshold=args.owner_high_threshold,
        low_threshold=args.owner_low_threshold,
        vote_window=args.owner_vote_window,
        min_votes=args.owner_min_votes,
        hold_frames=args.owner_hold_frames,
        crop_expansion=args.owner_reid_crop_expansion,
        crop_min_size=args.owner_reid_crop_min_size,
    )
    args._owner_profile = profile


def append_owner_frame_records(
    args: argparse.Namespace,
    frame_index: int,
    records: list[dict[str, Any]],
) -> None:
    if not hasattr(args, "_owner_frame_records"):
        args._owner_frame_records = []
    args._owner_frame_records.append(
        {
            "frame_index": frame_index,
            "faces": records,
        }
    )


def update_owner_keep_ids(
    args: argparse.Namespace,
    image: Image.Image,
    detections: list[FaceDetection],
    frame_index: int,
) -> set[str]:
    keep_track_ids = set(args.keep_track_ids)
    matcher = getattr(args, "_owner_matcher", None)
    if matcher is not None:
        owner_keep_ids, owner_records = matcher.match_frame(image, detections, frame_index)
        keep_track_ids.update(owner_keep_ids)
        append_owner_frame_records(args, frame_index, owner_records)
    args._active_keep_track_ids = keep_track_ids
    return keep_track_ids


def prepare_reference_prompt(args: argparse.Namespace) -> None:
    args._reference_prompt_text = ""
    args._reference_prompt_payload = None
    if args.reference_images:
        payload = build_reference_prompt_from_paths(
            image_paths=args.reference_images,
            mode=args.reference_prompt_mode,
            caption_model=args.reference_caption_model,
            device=args.reference_prompt_device,
            manual_tags=list(args.reference_manual_tags),
        )
        args._reference_prompt_payload = payload
        args._reference_prompt_text = str(payload["prompt"])
        if args.reference_prompt_json:
            write_reference_prompt(payload, args.reference_prompt_json, args.reference_prompt_text_file)
        elif args.reference_prompt_text_file:
            write_reference_prompt(payload, None, args.reference_prompt_text_file)
        return

    if args.reference_prompt_json:
        payload = load_reference_prompt(args.reference_prompt_json)
        args._reference_prompt_payload = payload
        args._reference_prompt_text = str(payload["prompt"])


def prepare_reference_face_bank(args: argparse.Namespace) -> None:
    args._reference_face_bank = None
    if not args.reference_face_images:
        return
    if args.inpaint_scope != "face-crop":
        warnings.warn(
            "--reference-face-images is only applied with --inpaint-scope face-crop.",
            RuntimeWarning,
        )
        return
    args._reference_face_bank = ReferenceFaceBank(
        image_paths=args.reference_face_images,
        crop_mode=args.reference_face_crop_mode,
        target_expansion=args.reference_face_target_expansion,
        feather=args.reference_face_feather,
        opacity=args.reference_face_opacity,
    )


def prepare_reference_identity_condition(args: argparse.Namespace) -> None:
    args._reference_identity_condition = None
    if not args.reference_identity_images:
        return
    if args.inpaint_scope != "face-crop":
        warnings.warn(
            "--reference-identity-images is designed for --inpaint-scope face-crop.",
            RuntimeWarning,
        )
    analyzer = InsightFaceReferenceAnalyzer(
        model_name=args.reference_face_model,
        model_root=args.reference_face_model_root,
        providers=args.reference_face_providers or ("CPUExecutionProvider",),
        det_size=args.reference_face_det_size,
    )
    condition = route_reference_identity(
        image_paths=args.reference_identity_images,
        route=args.reference_route,
        analyzer=analyzer,
        min_confidence=args.reference_human_threshold,
        human_min_ratio=args.reference_human_min_ratio,
        ip_adapter_sheet_size=args.ip_adapter_sheet_size,
        ip_adapter_max_images=args.ip_adapter_max_images,
    )
    args._reference_identity_condition = condition
    args._reference_route = condition.route
    if condition.is_ip_adapter:
        default_char_prompt = "a realistic character face matching the reference image, natural lighting, detailed face"
        if args.reference_character_prompt == default_char_prompt:
            try:
                print("Auto-captioning character/animal reference image(s) for prompt generation...")
                payload = build_reference_prompt_from_paths(
                    image_paths=args.reference_identity_images,
                    mode="caption",
                    caption_model=args.reference_caption_model,
                    device=args.reference_prompt_device,
                )
                if payload.get("captions"):
                    args.reference_character_prompt = ", ".join(payload["captions"]) + ", natural lighting"
                else:
                    args.reference_character_prompt = payload["prompt"]
                print(f"--> Auto-detected character prompt: {args.reference_character_prompt}")
            except Exception as e:
                warnings.warn(f"Failed to auto-caption reference: {e}. Using default character prompt.", RuntimeWarning)

        if args.mask_mode == "auto":
            args.mask_mode = "bbox"
        if args.mask_expansion == 1.35:
            args.mask_expansion = 1.0
        if args.mask_y_shift == -0.04:
            args.mask_y_shift = 0.0


def prepare_replacement_preset(args: argparse.Namespace) -> None:
    if args.replacement_preset != "fire":
        return
    if args.fire_force_bbox and args.mask_mode == "auto":
        args.mask_mode = "bbox"
    if args.fire_force_bbox and args.mask_expansion == 1.35:
        args.mask_expansion = 1.08
    if args.fire_force_bbox and args.mask_y_shift == -0.04:
        args.mask_y_shift = 0.0
    if args.mask_dilation == 9:
        args.mask_dilation = 20
    if args.mask_blur == 15:
        args.mask_blur = 20
    if args.strength == 0.98:
        args.strength = 1.0
    if args.guidance_scale == 5.0:
        args.guidance_scale = 6.0


def process_video(args: argparse.Namespace, detections_by_frame: dict[int, list[FaceDetection]]) -> None:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total = frame_count if args.max_frames <= 0 else min(frame_count, args.max_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*args.video_codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    pipe = None
    device = "cpu"
    if not args.mask_preview:
        pipe, device = load_pipeline(args)

    frame_index = 0
    pbar = tqdm(total=total if total > 0 else None, desc="frames")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            image = pil_from_bgr(frame)
            detections = detections_for_frame(
                detections_by_frame,
                frame_index=frame_index,
                hold_last_detections=args.hold_last_detections,
            )
            keep_track_ids = update_owner_keep_ids(args, image, detections, frame_index)
            mask = build_face_mask(
                image=image,
                image_size=image.size,
                detections=detections,
                keep_track_ids=keep_track_ids,
                anonymize_track_ids=args.anonymize_track_ids,
                anonymize_untracked=args.anonymize_untracked,
                mask_mode=args.mask_mode,
                mask_fallback=args.mask_fallback,
                mask_expansion=args.mask_expansion,
                mask_y_shift=args.mask_y_shift,
                mask_dilation=args.mask_dilation,
                mask_blur=args.mask_blur,
                args=args,
            )
            if args.mask_preview:
                output_image = overlay_mask(image, mask)
            elif args.inpaint_scope == "face-crop":
                output_image = run_face_crop_inpaint(pipe, device, image, detections, frame_index, args)
            else:
                identity = choose_identity(None, args.identity_bank_entries, 0)
                seed = seed_for_context(
                    base_seed=args.seed,
                    frame_index=frame_index,
                    track_id=None,
                    face_index=0,
                    identity=identity,
                    seed_strategy=args.seed_strategy,
                    vary_seed_per_frame=args.vary_seed_per_frame,
                )
                output_image = inpaint_or_fallback(
                    pipe=pipe,
                    device=device,
                    image=image,
                    mask=mask,
                    seed=seed,
                    identity=identity,
                    frame_index=frame_index,
                    detection=None,
                    face_index=0,
                    crop_box=None,
                    args=args,
                )

            writer.write(bgr_from_pil(output_image))
            frame_index += 1
            pbar.update(1)
    finally:
        pbar.close()
        capture.release()
        writer.release()


def input_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Unsupported input extension: {suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anonymize selected tracked faces with SDXL inpainting.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image or video.")
    parser.add_argument("--output", type=Path, help="Output image or video.")
    parser.add_argument("--detections", type=Path, required=True, help="YOLO/tracker detections JSON.")
    parser.add_argument("--bbox-format", choices=("xyxy", "xywh"), default="xyxy")
    parser.add_argument("--keep-track-ids", type=parse_id_list, default=set())
    parser.add_argument("--anonymize-track-ids", type=parse_id_list, default=set())
    parser.add_argument(
        "--anonymize-untracked",
        action="store_true",
        help="Anonymize detections without a track_id when no explicit anonymize IDs are set.",
    )
    parser.add_argument(
        "--hold-last-detections",
        action="store_true",
        help="Reuse the most recent detection set for frames missing JSON detections.",
    )
    parser.add_argument(
        "--owner-crops-dir",
        type=Path,
        help="Export source-frame face crops and a manifest for UI owner selection.",
    )
    parser.add_argument(
        "--owner-profile",
        type=Path,
        help=(
            "Load an owner embedding profile for Re-ID. With --owner-face-index, "
            "create/overwrite this profile from the selected source crop."
        ),
    )
    parser.add_argument(
        "--owner-face-index",
        type=int,
        help="Zero-based face index from --owner-source-frame to keep as the owner.",
    )
    parser.add_argument("--owner-source-frame", type=int, default=0)
    parser.add_argument("--owner-crop-expansion", type=float, default=1.45)
    parser.add_argument("--owner-crop-min-size", type=int, default=192)
    parser.add_argument("--owner-reid-crop-expansion", type=float, default=1.35)
    parser.add_argument("--owner-reid-crop-min-size", type=int, default=160)
    parser.add_argument("--owner-high-threshold", type=float, default=0.55)
    parser.add_argument("--owner-low-threshold", type=float, default=0.35)
    parser.add_argument("--owner-vote-window", type=int, default=10)
    parser.add_argument("--owner-min-votes", type=int, default=3)
    parser.add_argument("--owner-hold-frames", type=int, default=12)
    parser.add_argument("--face-recognition-model", default="buffalo_l")
    parser.add_argument(
        "--face-recognition-providers",
        type=parse_csv_tuple,
        default=("CPUExecutionProvider",),
        help="Comma-separated ONNX Runtime providers for InsightFace.",
    )
    parser.add_argument("--face-recognition-det-size", type=int, default=640)
    parser.add_argument(
        "--include-owner-embedding-in-report",
        action="store_true",
        help="Include biometric owner embedding values in --report-json.",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to use for image inputs.")
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "segmentation", "landmark", "ellipse", "bbox", "sam"),
        default="auto",
        help="Prefer segmentation polygons, landmark hulls, bbox ellipse/rectangle masks, or SAM box-prompt masks.",
    )
    parser.add_argument(
        "--mask-fallback",
        choices=("ellipse", "none"),
        default="ellipse",
        help="Fallback when selected mask metadata is missing.",
    )
    parser.add_argument("--mask-expansion", type=float, default=1.35)
    parser.add_argument("--mask-y-shift", type=float, default=-0.04)
    parser.add_argument("--mask-dilation", type=int, default=9)
    parser.add_argument("--mask-blur", type=int, default=15)
    parser.add_argument("--mask-preview", action="store_true", help="Save a red mask overlay instead of running SDXL.")
    parser.add_argument("--save-mask", type=Path, help="Optional path for the grayscale inpaint mask.")
    parser.add_argument("--sam-model-id", default=DEFAULT_SAM_MODEL_ID)
    parser.add_argument("--sam-device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--sam-mask-threshold", type=float, default=0.0)
    parser.add_argument(
        "--sam-local-files-only",
        action="store_true",
        help="Load the SAM model only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--sam-multimask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask SAM for multiple masks and keep the highest-IoU candidate.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--variant", default=None, help="Diffusers variant (e.g. fp16)")
    parser.add_argument(
        "--scheduler",
        choices=("default", "dpmpp_2m_karras", "euler_a"),
        default="dpmpp_2m_karras",
    )
    parser.add_argument(
        "--controlnet",
        choices=("none", "canny", "depth"),
        default="none",
        help="Optional SDXL ControlNet guidance for face crops/full-frame inpainting.",
    )
    parser.add_argument("--controlnet-model-id", help="Override the default ControlNet checkpoint.")
    parser.add_argument("--controlnet-scale", type=float, default=0.55)
    parser.add_argument("--controlnet-guess-mode", action="store_true")
    parser.add_argument("--controlnet-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--canny-low", type=int, default=80)
    parser.add_argument("--canny-high", type=int, default=180)
    parser.add_argument("--depth-estimator-model", default="Intel/dpt-hybrid-midas")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--replacement-preset",
        choices=("none", "fire"),
        default="none",
        help="Use a non-identity anonymization preset. fire covers the masked area with flames/smoke.",
    )
    parser.add_argument("--fire-prompt", default=DEFAULT_FIRE_PROMPT)
    parser.add_argument("--fire-negative-prompt", default=DEFAULT_FIRE_NEGATIVE_PROMPT)
    parser.add_argument(
        "--fire-prepaint",
        choices=("none", "white", "black", "gray"),
        default="white",
        help="Fill the masked input area before diffusion so the model does not reconstruct the original face.",
    )
    parser.add_argument(
        "--fire-prepaint-region",
        choices=("bbox", "mask"),
        default="bbox",
        help="Use the raw detection bbox or the final diffusion mask for the fire prepaint fill.",
    )
    parser.add_argument(
        "--fire-prepaint-bbox-expansion",
        type=float,
        default=1.0,
        help="Expansion for bbox-only fire prepaint. Keep at 1.0 to blank only the original bbox.",
    )
    parser.add_argument(
        "--fire-force-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --replacement-preset fire is active, make auto masks use the detection bbox.",
    )
    parser.add_argument(
        "--reference-images",
        type=Path,
        nargs="+",
        help="Reference images to summarize into additional non-identifying prompt traits.",
    )
    parser.add_argument(
        "--reference-prompt-json",
        type=Path,
        help="Save a generated reference prompt JSON, or load one when --reference-images is omitted.",
    )
    parser.add_argument("--reference-prompt-text-file", type=Path)
    parser.add_argument(
        "--reference-prompt-mode",
        choices=("heuristic", "caption", "auto"),
        default="heuristic",
        help="heuristic avoids model downloads; caption uses an image captioning model.",
    )
    parser.add_argument("--reference-caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--reference-prompt-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--reference-manual-tags",
        type=parse_csv_tuple,
        default=(),
        help="Comma-separated traits to append to the reference prompt.",
    )
    parser.add_argument(
        "--reference-face-images",
        type=Path,
        nargs="+",
        help="Reference face images to feather-blend into each target crop before inpainting.",
    )
    parser.add_argument(
        "--reference-face-crop-mode",
        choices=("auto", "center", "full"),
        default="auto",
        help="How to crop the reference face before fitting it into the target bbox.",
    )
    parser.add_argument("--reference-face-target-expansion", type=float, default=1.12)
    parser.add_argument("--reference-face-feather", type=int, default=18)
    parser.add_argument("--reference-face-opacity", type=float, default=0.92)
    parser.add_argument(
        "--reference-identity-images",
        type=Path,
        nargs="+",
        help="Reference images routed to InstantID for human faces or IP-Adapter for characters/animals.",
    )
    parser.add_argument(
        "--reference-route",
        choices=("auto", REFERENCE_ROUTE_INSTANTID, REFERENCE_ROUTE_IP_ADAPTER),
        default="auto",
        help="auto uses InsightFace detection to choose InstantID or IP-Adapter.",
    )
    parser.add_argument("--reference-human-threshold", type=float, default=0.55)
    parser.add_argument("--reference-human-min-ratio", type=float, default=0.5)
    parser.add_argument("--reference-target-face-threshold", type=float, default=0.35)
    _curr_dir = Path(__file__).resolve().parent
    _default_ref_root = (
        _curr_dir.parent if (_curr_dir.parent / "models").exists()
        else (_curr_dir.parent.parent if (_curr_dir.parent.parent / "models").exists() else Path("."))
    )
    parser.add_argument("--reference-face-model", default="antelopev2")
    parser.add_argument("--reference-face-model-root", type=Path, default=_default_ref_root)
    parser.add_argument(
        "--reference-face-providers",
        type=parse_csv_tuple,
        default=("CPUExecutionProvider",),
        help="Comma-separated ONNX Runtime providers for reference routing.",
    )
    parser.add_argument("--reference-face-det-size", type=int, default=640)
    parser.add_argument("--instantid-base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--instantid-repo-id", default="InstantX/InstantID")
    parser.add_argument("--instantid-controlnet-model", default="InstantX/InstantID")
    parser.add_argument("--instantid-controlnet-subfolder", default="ControlNetModel")
    parser.add_argument("--instantid-adapter-path", type=Path)
    parser.add_argument("--instantid-adapter-filename", default="ip-adapter.bin")
    parser.add_argument("--instantid-pipeline-dir", type=Path)
    parser.add_argument("--instantid-controlnet-scale", type=float, default=0.8)
    parser.add_argument("--instantid-ip-adapter-scale", type=float, default=0.8)
    parser.add_argument("--instantid-prompt", default="a face")
    parser.add_argument(
        "--instantid-negative-prompt",
        default="lowres, bad anatomy, worst quality, low quality, blurry, deformed",
    )
    parser.add_argument("--ip-adapter-model", default="h94/IP-Adapter")
    parser.add_argument("--ip-adapter-subfolder", default="sdxl_models")
    parser.add_argument("--ip-adapter-weight-name", default="ip-adapter-plus_sdxl_vit-h.safetensors")
    parser.add_argument("--ip-adapter-image-encoder-model")
    parser.add_argument("--ip-adapter-image-encoder-subfolder", default="models/image_encoder")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.75)
    parser.add_argument("--ip-adapter-sheet-size", type=int, default=512)
    parser.add_argument("--ip-adapter-max-images", type=int, default=4)
    parser.add_argument(
        "--reference-character-prompt",
        default="a realistic character face matching the reference image, natural lighting, detailed face",
    )
    parser.add_argument("--identity-bank", type=Path, help="JSON bank of synthetic identities and optional LoRAs.")
    parser.add_argument("--lora", help="Optional synthetic identity LoRA path or Hugging Face repo.")
    parser.add_argument("--lora-adapter-name", default="synthetic_identity")
    parser.add_argument("--lora-weight", type=float, default=0.8)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--cpu-offload", action="store_true", help="Use Diffusers CPU offload on CUDA.")
    parser.add_argument("--attention-slicing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--inpaint-scope",
        choices=("face-crop", "full-frame"),
        default="face-crop",
        help="Use face-crop for better small-face quality or full-frame for one-pass generation.",
    )
    parser.add_argument("--crop-expansion", type=float, default=2.4)
    parser.add_argument("--crop-min-size", type=int, default=512)
    parser.add_argument("--max-side", type=int, default=1024, help="Resize long side for the SDXL working image.")
    parser.add_argument("--strength", type=float, default=0.98)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--fallback-mode", choices=("blur", "pixelate", "none"), default="blur")
    parser.add_argument("--fallback-blur-radius", type=float, default=18.0)
    parser.add_argument("--fallback-pixel-size", type=int, default=12)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.0003)
    parser.add_argument("--min-mean-delta", type=float, default=2.0)
    parser.add_argument("--report-json", type=Path, help="Optional quality/fallback report path.")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--seed-strategy",
        choices=("global", "track", "frame", "frame-track"),
        default="track",
        help="How to derive deterministic seeds from --seed for video/crop inpainting.",
    )
    parser.add_argument("--vary-seed-per-frame", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="For video demos, stop after this many frames.")
    parser.add_argument("--video-codec", default="mp4v")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_id == "fallback":
        args.model_id = FALLBACK_MODEL_ID
    args.identity_bank_entries = load_identity_bank(args.identity_bank)
    args._report_records = []
    args._owner_frame_records = []
    args._active_keep_track_ids = set(args.keep_track_ids)
    detections_by_frame = load_detections(args.detections, bbox_format=args.bbox_format)
    kind = input_kind(args.input)
    maybe_export_owner_crops(args, detections_by_frame, kind)
    crop_export_only = (
        args.owner_crops_dir
        and args.output is None
        and args.owner_face_index is None
        and args.owner_profile is None
    )
    if crop_export_only:
        return
    if args.output is None:
        raise ValueError("--output is required unless you are only exporting --owner-crops-dir.")
    prepare_reference_prompt(args)
    prepare_reference_face_bank(args)
    prepare_reference_identity_condition(args)
    prepare_replacement_preset(args)
    prepare_owner_matcher(args, detections_by_frame, kind)
    if kind == "image":
        process_image(args, detections_by_frame)
    else:
        process_video(args, detections_by_frame)
    write_quality_report(args)


if __name__ == "__main__":
    main()

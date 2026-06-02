"""Reference-face conditioning for stronger synthetic identity preservation.

This is a lightweight pre-inpaint step: crop a reference face, resize it into
the target face region, feather-blend it, then let SDXL harmonize the result.
Use only synthetic, licensed, or consented reference identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def _bbox_of(detection: Any) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in detection.bbox)  # type: ignore[attr-defined,return-value]


def _expanded_box(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    expansion: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    new_w = box_w * expansion
    new_h = box_h * expansion
    left = max(0, int(round(cx - new_w * 0.5)))
    top = max(0, int(round(cy - new_h * 0.5)))
    right = min(width, int(round(cx + new_w * 0.5)))
    bottom = min(height, int(round(cy + new_h * 0.5)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _largest_haar_face(image: Image.Image) -> tuple[int, int, int, int] | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        return None
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    return _expanded_box((x, y, x + w, y + h), image.width, image.height, expansion=1.55)


def _center_square(image: Image.Image) -> tuple[int, int, int, int]:
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return left, top, left + side, top + side


@dataclass(frozen=True)
class ReferenceFace:
    path: Path
    crop: Image.Image
    source_box: tuple[int, int, int, int]


class ReferenceFaceBank:
    def __init__(
        self,
        image_paths: list[Path],
        crop_mode: str = "auto",
        target_expansion: float = 1.12,
        feather: int = 18,
        opacity: float = 0.92,
    ) -> None:
        if not image_paths:
            raise ValueError("At least one reference face image is required.")
        self.crop_mode = crop_mode
        self.target_expansion = target_expansion
        self.feather = max(0, int(feather))
        self.opacity = min(max(float(opacity), 0.0), 1.0)
        self.faces = [self._load_reference(path) for path in image_paths]

    def _load_reference(self, path: Path) -> ReferenceFace:
        image = Image.open(path).convert("RGB")
        if self.crop_mode == "full":
            source_box = (0, 0, image.width, image.height)
        elif self.crop_mode == "center":
            source_box = _center_square(image)
        else:
            source_box = _largest_haar_face(image) or _center_square(image)
        return ReferenceFace(path=path, crop=image.crop(source_box), source_box=source_box)

    def choose(self, track_id: str | None, face_index: int) -> ReferenceFace:
        key = str(track_id) if track_id is not None else str(face_index)
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, byteorder="big", signed=False) % len(self.faces)
        return self.faces[index]

    def apply_to_crop(
        self,
        crop: Image.Image,
        detection: Any,
        track_id: str | None,
        face_index: int,
    ) -> Image.Image:
        target_box = _expanded_box(
            _bbox_of(detection),
            width=crop.width,
            height=crop.height,
            expansion=self.target_expansion,
        )
        if target_box is None:
            return crop

        reference = self.choose(track_id, face_index)
        left, top, right, bottom = target_box
        target_size = (right - left, bottom - top)
        if target_size[0] <= 0 or target_size[1] <= 0:
            return crop

        reference_patch = ImageOps.fit(
            reference.crop,
            target_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.42),
        )
        mask = Image.new("L", target_size, 0)
        draw = ImageDraw.Draw(mask)
        inset_x = max(1, int(target_size[0] * 0.04))
        inset_y = max(1, int(target_size[1] * 0.03))
        draw.ellipse(
            (inset_x, inset_y, target_size[0] - inset_x, target_size[1] - inset_y),
            fill=int(255 * self.opacity),
        )
        if self.feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.feather))

        output = crop.copy()
        original_region = output.crop(target_box)
        blended_region = Image.composite(reference_patch, original_region, mask)
        output.paste(blended_region, target_box)
        return output

"""Owner face Re-ID helpers for selective anonymization.

The YOLO/tracker side supplies face boxes. This module turns one selected
source crop into a face embedding, then uses that embedding to recover the
owner track across later frames.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


def _bbox_of(detection: Any) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in detection.bbox)  # type: ignore[attr-defined,return-value]


def _track_id_of(detection: Any) -> str | None:
    track_id = getattr(detection, "track_id", None)
    return str(track_id) if track_id is not None else None


def _confidence_of(detection: Any) -> float | None:
    confidence = getattr(detection, "confidence", None)
    return float(confidence) if confidence is not None else None


def normalize_embedding(embedding: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(embedding), dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("Face embedding has zero length.")
    return values / norm


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = normalize_embedding(left)
    right_values = normalize_embedding(right)
    return float(np.dot(left_values, right_values))


def expanded_crop_box(
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


def crop_detection(
    image: Image.Image,
    detection: Any,
    expansion: float,
    min_size: int,
) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    crop_box = expanded_crop_box(
        _bbox_of(detection),
        width=image.width,
        height=image.height,
        expansion=expansion,
        min_size=min_size,
    )
    if crop_box is None:
        return None
    return image.crop(crop_box), crop_box


@dataclass(frozen=True)
class OwnerProfile:
    source_frame: int
    face_index: int
    embedding_model: str
    embedding: tuple[float, ...]
    bbox: tuple[float, float, float, float] | None = None
    track_id: str | None = None
    crop_box: tuple[int, int, int, int] | None = None

    @classmethod
    def from_json(cls, path: Path) -> "OwnerProfile":
        payload = json.loads(path.read_text(encoding="utf-8"))
        owner = payload.get("owner", payload) if isinstance(payload, dict) else payload
        if not isinstance(owner, dict):
            raise ValueError("Owner profile must be a JSON object.")
        if "embedding" not in owner:
            raise ValueError("Owner profile is missing an embedding.")
        return cls(
            source_frame=int(owner.get("source_frame", 0)),
            face_index=int(owner.get("face_index", 0)),
            embedding_model=str(owner.get("embedding_model", "insightface/buffalo_l")),
            embedding=tuple(float(value) for value in owner["embedding"]),
            bbox=tuple(float(value) for value in owner["bbox"]) if owner.get("bbox") else None,
            track_id=str(owner["track_id"]) if owner.get("track_id") is not None else None,
            crop_box=tuple(int(value) for value in owner["crop_box"]) if owner.get("crop_box") else None,
        )

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_frame": self.source_frame,
            "face_index": self.face_index,
            "embedding_model": self.embedding_model,
            "track_id": self.track_id,
            "bbox": list(self.bbox) if self.bbox else None,
            "crop_box": list(self.crop_box) if self.crop_box else None,
        }
        if include_embedding:
            payload["embedding"] = [round(float(value), 8) for value in self.embedding]
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"owner": self.to_dict(include_embedding=True)}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class InsightFaceEmbedder:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        det_size: int = 640,
    ) -> None:
        self.model_name = model_name
        self.providers = providers
        self.det_size = det_size
        self._app: Any | None = None

    @property
    def profile_model_name(self) -> str:
        return f"insightface/{self.model_name}"

    def _ensure_app(self) -> Any:
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "Owner Re-ID requires insightface. Install requirements.txt or add "
                "'insightface' and 'onnxruntime' to the environment."
            ) from exc

        ctx_id = 0 if any(provider != "CPUExecutionProvider" for provider in self.providers) else -1
        app = FaceAnalysis(name=self.model_name, providers=list(self.providers))
        app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        self._app = app
        return app

    def embed(self, image: Image.Image) -> tuple[float, ...] | None:
        app = self._ensure_app()
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            return None

        def face_area(face: Any) -> float:
            x1, y1, x2, y2 = (float(value) for value in face.bbox)
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        face = max(faces, key=face_area)
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = normalize_embedding(getattr(face, "embedding"))
        else:
            embedding = normalize_embedding(embedding)
        return tuple(float(value) for value in embedding)


def build_owner_profile(
    image: Image.Image,
    detections: list[Any],
    face_index: int,
    embedder: InsightFaceEmbedder,
    source_frame: int,
    crop_expansion: float,
    crop_min_size: int,
) -> OwnerProfile:
    if face_index < 0 or face_index >= len(detections):
        raise ValueError(
            f"owner face_index {face_index} is out of range for frame {source_frame}; "
            f"available faces: {len(detections)}"
        )
    detection = detections[face_index]
    cropped = crop_detection(
        image,
        detection,
        expansion=crop_expansion,
        min_size=crop_min_size,
    )
    if cropped is None:
        raise ValueError(f"Could not crop owner face_index {face_index}.")
    crop, crop_box = cropped
    embedding = embedder.embed(crop)
    if embedding is None:
        raise RuntimeError(
            "InsightFace could not detect a recognizable face in the selected owner crop. "
            "Try a clearer source frame or increase --owner-crop-expansion."
        )
    return OwnerProfile(
        source_frame=source_frame,
        face_index=face_index,
        embedding_model=embedder.profile_model_name,
        embedding=embedding,
        bbox=_bbox_of(detection),
        track_id=_track_id_of(detection),
        crop_box=crop_box,
    )


def export_face_crops(
    image: Image.Image,
    detections: list[Any],
    output_dir: Path,
    source_frame: int,
    crop_expansion: float,
    crop_min_size: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for face_index, detection in enumerate(detections):
        cropped = crop_detection(
            image,
            detection,
            expansion=crop_expansion,
            min_size=crop_min_size,
        )
        if cropped is None:
            continue
        crop, crop_box = cropped
        track_id = _track_id_of(detection)
        track_label = f"track_{track_id}" if track_id is not None else "untracked"
        filename = f"frame_{source_frame:06d}_face_{face_index:03d}_{track_label}.jpg"
        crop.save(output_dir / filename, quality=92)
        entries.append(
            {
                "face_index": face_index,
                "track_id": track_id,
                "bbox": list(_bbox_of(detection)),
                "crop_box": list(crop_box),
                "confidence": _confidence_of(detection),
                "crop_path": filename,
            }
        )

    manifest = {
        "source_frame": source_frame,
        "faces": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


@dataclass
class TrackOwnerState:
    similarities: deque[float] = field(default_factory=deque)
    is_owner: bool = False
    hold_until_frame: int = -1


class OwnerMatcher:
    def __init__(
        self,
        profile: OwnerProfile,
        embedder: InsightFaceEmbedder,
        high_threshold: float,
        low_threshold: float,
        vote_window: int,
        min_votes: int,
        hold_frames: int,
        crop_expansion: float,
        crop_min_size: int,
    ) -> None:
        self.profile = profile
        self.embedder = embedder
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.vote_window = max(1, vote_window)
        self.min_votes = max(1, min_votes)
        self.hold_frames = max(0, hold_frames)
        self.crop_expansion = crop_expansion
        self.crop_min_size = crop_min_size
        self.track_states: dict[str, TrackOwnerState] = {}

    def _state_for(self, track_id: str) -> TrackOwnerState:
        if track_id not in self.track_states:
            self.track_states[track_id] = TrackOwnerState()
        return self.track_states[track_id]

    def _update_track_state(
        self,
        track_id: str,
        similarity: float,
        frame_index: int,
    ) -> bool:
        state = self._state_for(track_id)
        state.similarities.append(similarity)
        while len(state.similarities) > self.vote_window:
            state.similarities.popleft()

        owner_votes = sum(value >= self.high_threshold for value in state.similarities)
        mean_similarity = float(np.mean(state.similarities))
        if similarity >= self.high_threshold or owner_votes >= self.min_votes:
            state.is_owner = True
            state.hold_until_frame = frame_index + self.hold_frames
        elif similarity <= self.low_threshold and mean_similarity <= self.low_threshold and owner_votes == 0:
            state.is_owner = False
            state.hold_until_frame = -1
        elif state.is_owner and frame_index <= state.hold_until_frame:
            state.is_owner = True

        return state.is_owner

    def match_frame(
        self,
        image: Image.Image,
        detections: list[Any],
        frame_index: int,
    ) -> tuple[set[str], list[dict[str, Any]]]:
        keep_track_ids: set[str] = set()
        records: list[dict[str, Any]] = []
        for face_index, detection in enumerate(detections):
            track_id = _track_id_of(detection)
            similarity: float | None = None
            is_owner = False
            error: str | None = None
            cropped = crop_detection(
                image,
                detection,
                expansion=self.crop_expansion,
                min_size=self.crop_min_size,
            )
            if cropped is None:
                error = "crop_failed"
            else:
                crop, _ = cropped
                embedding = self.embedder.embed(crop)
                if embedding is None:
                    error = "embedding_failed"
                else:
                    similarity = cosine_similarity(self.profile.embedding, embedding)

            if similarity is not None:
                if track_id is not None:
                    is_owner = self._update_track_state(track_id, similarity, frame_index)
                    if is_owner:
                        keep_track_ids.add(track_id)
                else:
                    is_owner = similarity >= self.high_threshold

            records.append(
                {
                    "face_index": face_index,
                    "track_id": track_id,
                    "bbox": list(_bbox_of(detection)),
                    "owner_similarity": round(similarity, 6) if similarity is not None else None,
                    "is_owner": is_owner,
                    "error": error,
                }
            )
        return keep_track_ids, records

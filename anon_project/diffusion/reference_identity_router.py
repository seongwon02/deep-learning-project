"""Route reference images to InstantID or IP-Adapter conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


REFERENCE_ROUTE_INSTANTID = "instantid"
REFERENCE_ROUTE_IP_ADAPTER = "ip-adapter"


def _face_value(face: Any, key: str, default: Any = None) -> Any:
    if isinstance(face, dict):
        return face.get(key, default)
    return getattr(face, key, default)


def _face_area(face: Any) -> float:
    bbox = _face_value(face, "bbox")
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _det_score(face: Any) -> float:
    score = _face_value(face, "det_score", _face_value(face, "score", 1.0))
    return float(score if score is not None else 1.0)


def _embedding(face: Any) -> np.ndarray | None:
    embedding = _face_value(face, "embedding")
    if embedding is None:
        embedding = _face_value(face, "normed_embedding")
    if embedding is None:
        return None
    return np.asarray(embedding, dtype=np.float32)


def _kps(face: Any) -> np.ndarray | None:
    keypoints = _face_value(face, "kps")
    if keypoints is None:
        keypoints = _face_value(face, "landmark_2d_106")
    if keypoints is None:
        return None
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape[0] < 5:
        return None
    return keypoints[:5, :2]


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bbox_of(detection: Any) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in detection.bbox)  # type: ignore[attr-defined,return-value]


@dataclass(frozen=True)
class ReferenceFaceAnalysis:
    path: Path
    score: float
    bbox: tuple[float, float, float, float]
    kps: np.ndarray
    embedding: np.ndarray


@dataclass
class ReferenceIdentityCondition:
    route: str
    image_paths: list[Path]
    reference_images: list[Image.Image]
    face_analyses: list[ReferenceFaceAnalysis]
    face_embedding: np.ndarray | None = None
    ip_adapter_image: Image.Image | None = None
    analyzer: "InsightFaceReferenceAnalyzer | None" = None

    @property
    def is_instantid(self) -> bool:
        return self.route == REFERENCE_ROUTE_INSTANTID

    @property
    def is_ip_adapter(self) -> bool:
        return self.route == REFERENCE_ROUTE_IP_ADAPTER


class InsightFaceReferenceAnalyzer:
    def __init__(
        self,
        model_name: str = "antelopev2",
        model_root: Path | str = ".",
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        det_size: int = 640,
    ) -> None:
        self.model_name = model_name
        self.model_root = str(model_root)
        self.providers = providers
        self.det_size = det_size
        self._app: Any | None = None

    def _ensure_app(self) -> Any:
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("Reference routing requires insightface.") from exc

        ctx_id = 0 if any(provider != "CPUExecutionProvider" for provider in self.providers) else -1
        app = FaceAnalysis(
            name=self.model_name,
            root=self.model_root,
            providers=list(self.providers),
        )
        app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        self._app = app
        return app

    def detect_faces(self, image: Image.Image) -> list[Any]:
        app = self._ensure_app()
        faces = app.get(_pil_to_bgr(image))
        return sorted(faces, key=_face_area, reverse=True)

    def largest_valid_face(
        self,
        image: Image.Image,
        min_confidence: float,
    ) -> tuple[Any, float] | None:
        for face in self.detect_faces(image):
            score = _det_score(face)
            if score >= min_confidence and _embedding(face) is not None and _kps(face) is not None:
                return face, score
        return None

    def analyze_reference_image(
        self,
        path: Path,
        min_confidence: float,
    ) -> ReferenceFaceAnalysis | None:
        image = Image.open(path).convert("RGB")
        face_record = self.largest_valid_face(image, min_confidence=min_confidence)
        if face_record is None:
            return None
        face, score = face_record
        bbox = tuple(float(value) for value in _face_value(face, "bbox"))
        embedding = _embedding(face)
        keypoints = _kps(face)
        if embedding is None or keypoints is None:
            return None
        return ReferenceFaceAnalysis(
            path=path,
            score=score,
            bbox=bbox,  # type: ignore[arg-type]
            kps=keypoints,
            embedding=embedding,
        )

    def target_keypoints(
        self,
        image: Image.Image,
        detection: Any,
        min_confidence: float,
    ) -> np.ndarray:
        face_record = self.largest_valid_face(image, min_confidence=min_confidence)
        if face_record is not None:
            face, _ = face_record
            keypoints = _kps(face)
            if keypoints is not None:
                return keypoints

        landmarks = getattr(detection, "landmarks", None)
        if landmarks and len(landmarks) >= 5:
            return np.asarray(landmarks[:5], dtype=np.float32)
        return approximate_five_keypoints(_bbox_of(detection))


def approximate_five_keypoints(
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return np.asarray(
        [
            (x1 + width * 0.34, y1 + height * 0.38),
            (x1 + width * 0.66, y1 + height * 0.38),
            (x1 + width * 0.50, y1 + height * 0.55),
            (x1 + width * 0.38, y1 + height * 0.74),
            (x1 + width * 0.62, y1 + height * 0.74),
        ],
        dtype=np.float32,
    )


def draw_landmark_condition(
    image_size: tuple[int, int],
    keypoints: np.ndarray,
    radius: int = 4,
) -> Image.Image:
    width, height = image_size
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    points = [(float(x), float(y)) for x, y in keypoints[:5]]
    if len(points) >= 5:
        lines = [
            (points[0], points[2]),
            (points[1], points[2]),
            (points[2], points[3]),
            (points[2], points[4]),
            (points[3], points[4]),
        ]
        for start, end in lines:
            draw.line((start, end), fill=(255, 255, 255), width=max(1, radius // 2))
    colors = [(255, 80, 80), (80, 160, 255), (80, 255, 120), (255, 220, 80), (255, 120, 255)]
    for index, (x, y) in enumerate(points):
        color = colors[index % len(colors)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def average_embeddings(analyses: list[ReferenceFaceAnalysis]) -> np.ndarray:
    if not analyses:
        raise ValueError("Cannot average an empty list of face analyses.")
    embeddings = [analysis.embedding.astype(np.float32) for analysis in analyses]
    return np.mean(np.stack(embeddings, axis=0), axis=0)


def contact_sheet(images: list[Image.Image], cell_size: int = 512, max_images: int = 4) -> Image.Image:
    selected = images[:max(1, max_images)]
    columns = min(2, len(selected))
    rows = int(math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * cell_size, rows * cell_size), (0, 0, 0))
    for index, image in enumerate(selected):
        fitted = ImageOps.fit(image.convert("RGB"), (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        x = (index % columns) * cell_size
        y = (index // columns) * cell_size
        sheet.paste(fitted, (x, y))
    return sheet


def route_reference_identity(
    image_paths: list[Path],
    route: str,
    analyzer: InsightFaceReferenceAnalyzer,
    min_confidence: float,
    human_min_ratio: float,
    ip_adapter_sheet_size: int,
    ip_adapter_max_images: int,
) -> ReferenceIdentityCondition:
    if not image_paths:
        raise ValueError("At least one reference identity image is required.")

    reference_images = [Image.open(path).convert("RGB") for path in image_paths]
    analyses: list[ReferenceFaceAnalysis] = []
    if route in {"auto", REFERENCE_ROUTE_INSTANTID}:
        analyses = [
            analysis
            for path in image_paths
            if (analysis := analyzer.analyze_reference_image(path, min_confidence=min_confidence)) is not None
        ]

    human_ratio = len(analyses) / float(len(image_paths))
    if route == REFERENCE_ROUTE_INSTANTID or (route == "auto" and human_ratio >= human_min_ratio):
        if not analyses:
            raise RuntimeError("InstantID route requested, but no valid human reference face was detected.")
        return ReferenceIdentityCondition(
            route=REFERENCE_ROUTE_INSTANTID,
            image_paths=image_paths,
            reference_images=reference_images,
            face_analyses=analyses,
            face_embedding=average_embeddings(analyses),
            analyzer=analyzer,
        )

    return ReferenceIdentityCondition(
        route=REFERENCE_ROUTE_IP_ADAPTER,
        image_paths=image_paths,
        reference_images=reference_images,
        face_analyses=analyses,
        ip_adapter_image=contact_sheet(
            reference_images,
            cell_size=ip_adapter_sheet_size,
            max_images=ip_adapter_max_images,
        ),
        analyzer=analyzer,
    )

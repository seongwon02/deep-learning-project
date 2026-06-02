"""YOLO/트래커 탐지 JSON 파싱.

여러 포맷을 모두 frame_index -> [face dict] 형태로 정규화한다.
허용 포맷:
  - {"frames":[{"frame_index":0,"faces":[{"track_id":1,"bbox":[...],"keypoints":[...]}]}]}
  - {"frame_0":[{"id":1,"bbox":[...]}], "frame_5":[...]}   (기존 yolo_11_L 포맷)
  - [{...}, {...}]  (프레임 리스트)
face dict는 {"track_id": str|None, "bbox": (x1,y1,x2,y2), "keypoints": [(x,y),...]} 로 통일.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_int_frame_key(key: Any) -> int:
    digits = "".join(ch for ch in str(key) if ch.isdigit())
    return int(digits) if digits else 0


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, dict):
        if {"x1", "y1", "x2", "y2"} <= set(raw):
            return float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"])
        if {"x", "y", "w", "h"} <= set(raw):
            x, y, w, h = (float(raw[k]) for k in ("x", "y", "w", "h"))
            return x, y, x + w, y + h
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(float(v) for v in raw)  # type: ignore[return-value]
    return None


def _parse_keypoints(raw: Any) -> list[tuple[float, float]]:
    """[[x,y],...] 또는 [x,y,conf,...] 평면 배열 모두 허용. (x,y)만 추출."""
    if raw is None:
        return []
    pts: list[tuple[float, float]] = []
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(v, (int, float)) for v in raw):
        step = 3 if len(raw) % 3 == 0 else 2
        for i in range(0, len(raw) - 1, step):
            pts.append((float(raw[i]), float(raw[i + 1])))
        return pts
    if isinstance(raw, (list, tuple)):
        for p in raw:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
            elif isinstance(p, dict) and {"x", "y"} <= set(p):
                pts.append((float(p["x"]), float(p["y"])))
    return pts


def _face_track_id(raw: dict) -> str | None:
    for k in ("track_id", "trackId", "id", "track"):
        if k in raw and raw[k] is not None:
            return str(raw[k])
    return None


def _face_keypoints(raw: dict) -> list[tuple[float, float]]:
    for k in ("keypoints", "kps", "landmarks", "face_landmarks", "points"):
        if k in raw:
            return _parse_keypoints(raw[k])
    return []


def _iter_faces(frame_index: int, faces: Any, by_frame: dict) -> None:
    if not isinstance(faces, list):
        return
    for raw in faces:
        if not isinstance(raw, dict):
            continue
        bbox = None
        for k in ("bbox", "box", "xyxy", "bounds"):
            if k in raw:
                bbox = _parse_bbox(raw[k])
                break
        if bbox is None:
            bbox = _parse_bbox(raw)
        if bbox is None:
            continue
        by_frame.setdefault(frame_index, []).append({
            "track_id": _face_track_id(raw),
            "bbox": bbox,
            "keypoints": _face_keypoints(raw),
        })


def load_detections(path: Path) -> dict[int, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict]] = {}

    if isinstance(payload, dict) and "frames" in payload:
        frames = payload["frames"]
        if isinstance(frames, dict):
            for fk, faces in frames.items():
                _iter_faces(_to_int_frame_key(fk), faces, by_frame)
        else:
            for i, fr in enumerate(frames):
                if not isinstance(fr, dict):
                    continue
                fidx = fr.get("frame_index", fr.get("frame_idx", fr.get("frame", i)))
                faces = (fr.get("faces") or fr.get("detections")
                         or fr.get("objects") or fr.get("tracks") or [])
                _iter_faces(int(fidx), faces, by_frame)
        return by_frame

    if isinstance(payload, dict):  # {"frame_0":[...], ...}
        for fk, faces in payload.items():
            _iter_faces(_to_int_frame_key(fk), faces, by_frame)
        return by_frame

    if isinstance(payload, list):
        for i, rec in enumerate(payload):
            if not isinstance(rec, dict):
                continue
            if any(k in rec for k in ("faces", "detections", "objects", "tracks")):
                fidx = rec.get("frame_index", rec.get("frame", i))
                faces = (rec.get("faces") or rec.get("detections")
                         or rec.get("objects") or rec.get("tracks") or [])
                _iter_faces(int(fidx), faces, by_frame)
            else:
                _iter_faces(_to_int_frame_key(rec.get("frame_index", i)), [rec], by_frame)
        return by_frame

    raise ValueError("지원하지 않는 detections JSON 구조입니다.")


def detections_for_frame(by_frame: dict[int, list[dict]], idx: int, hold_last: bool) -> list[dict]:
    if idx in by_frame:
        return by_frame[idx]
    if not hold_last:
        return []
    prev = [k for k in by_frame if k <= idx]
    return by_frame[max(prev)] if prev else []


def all_track_ids(by_frame: dict[int, list[dict]]) -> set[str]:
    return {f["track_id"] for v in by_frame.values() for f in v if f["track_id"] is not None}
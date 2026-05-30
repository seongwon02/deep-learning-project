"""프레임 단위 비식별화 합성.

- 작은 얼굴(한 변 < min_face_size)은 스티커 대신 픽셀화/블러.
- 그 외에는 스티커를 얹는다. 키포인트가 있으면 두 눈 각도로 회전·정렬,
  없으면 bbox 중심에 정사각으로 넉넉히 배치.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class StickerConfig:
    """합성 동작을 조절하는 파라미터 묶음."""
    box_scale: float = 1.9       # bbox 한 변 대비 스티커 배율 (이마까지 덮도록 넉넉히)
    eye_scale: float = 3.2       # 키포인트 있을 때 눈 간격 대비 배율
    y_shift: float = -0.18       # 스티커 중심을 bbox 높이 대비 위로 이동(이마/머리 덮기), 음수=위로
    min_face_size: int = 40      # 이 픽셀(한 변) 미만은 블러
    blur_blocks: int = 8         # 작은 얼굴 픽셀화 블록 수(작을수록 더 뭉갬)


def pixelate_region(frame_bgr: np.ndarray, box: tuple[int, int, int, int], blocks: int = 8) -> None:
    x1, y1, x2, y2 = box
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)), interpolation=cv2.INTER_LINEAR)
    frame_bgr[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def eyes_angle_and_scale(kps: list[tuple[float, float]]) -> tuple[float, float] | None:
    """5점 키포인트(좌안,우안,코,좌입,우입)에서 회전각(deg)과 눈 간격 반환."""
    if len(kps) < 2:
        return None
    le, re = kps[0], kps[1]
    dx, dy = re[0] - le[0], re[1] - le[1]
    return math.degrees(math.atan2(dy, dx)), math.hypot(dx, dy)


def paste_sticker_rgba(base_rgba: Image.Image, sticker: Image.Image,
                       center: tuple[float, float], size: int, angle_deg: float = 0.0) -> None:
    """sticker를 size x size로 맞추고 angle 회전 후 center에 알파 합성(in-place).
    음수 좌표/경계 넘침을 수동으로 안전 처리한다."""
    s = sticker.resize((max(1, size), max(1, size)), Image.LANCZOS)
    if abs(angle_deg) > 0.5:
        s = s.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)  # 이미지 좌표계 보정

    cx, cy = center
    x = int(round(cx - s.width / 2))
    y = int(round(cy - s.height / 2))
    bw, bh = base_rgba.size

    sx1, sy1 = max(0, -x), max(0, -y)
    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(bw, x + s.width), min(bh, y + s.height)
    if dx2 <= dx1 or dy2 <= dy1:
        return
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)

    region = s.crop((sx1, sy1, sx2, sy2))
    base_region = base_rgba.crop((dx1, dy1, dx2, dy2))
    base_region.alpha_composite(region)
    base_rgba.paste(base_region, (dx1, dy1))


def _sticker_size_and_pose(face: dict, box: tuple[int, int, int, int],
                           cfg: StickerConfig) -> tuple[tuple[float, float], int, float]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    angle = 0.0

    kps = face.get("keypoints") or []
    info = eyes_angle_and_scale(kps)
    if info is not None and len(kps) >= 3:
        angle, eye_dist = info
        xs = [p[0] for p in kps[:5]]
        ys = [p[1] for p in kps[:5]]
        center = (sum(xs) / len(xs), sum(ys) / len(ys))
        size = max(int(eye_dist * cfg.eye_scale), int(max(bw, bh) * cfg.box_scale))
    else:
        # 키포인트 없음: bbox 중심을 위로 올려 이마/머리까지 덮고, 넉넉히 키운다
        center = (center[0], center[1] + bh * cfg.y_shift)
        size = int(max(bw, bh) * cfg.box_scale)

    return center, max(size, cfg.min_face_size), angle


def anonymize_frame(frame_bgr: np.ndarray, faces: list[dict], sticker: Image.Image,
                    keep_ids: set[str], cfg: StickerConfig) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    # 1) 작은 얼굴은 BGR 단계에서 먼저 블러, 큰 얼굴은 스티커 대상으로 모음
    sticker_targets: list[tuple[dict, tuple[int, int, int, int]]] = []
    for face in faces:
        tid = face["track_id"]
        if tid is not None and tid in keep_ids:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in face["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if min(x2 - x1, y2 - y1) < cfg.min_face_size:
            pixelate_region(frame_bgr, (x1, y1, x2, y2), blocks=cfg.blur_blocks)
        else:
            sticker_targets.append((face, (x1, y1, x2, y2)))

    if not sticker_targets:
        return frame_bgr

    # 2) 스티커는 RGBA 캔버스에서 알파 합성
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    for face, box in sticker_targets:
        center, size, angle = _sticker_size_and_pose(face, box, cfg)
        paste_sticker_rgba(base, sticker, center, size, angle_deg=angle)

    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
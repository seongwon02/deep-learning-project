"""이미지/영상 처리 오케스트레이션."""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from .compositing import StickerConfig, anonymize_frame
from .detections import detections_for_frame


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def input_kind(path: Path) -> str:
    s = path.suffix.lower()
    if s in IMAGE_EXTS:
        return "image"
    if s in VIDEO_EXTS:
        return "video"
    raise ValueError(f"지원하지 않는 확장자: {s}")


def process_image(input_path: Path, output_path: Path, by_frame, sticker,
                  keep_ids: set[str], cfg: StickerConfig,
                  frame_index: int = 0, hold_last: bool = False) -> None:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f"이미지를 열 수 없습니다: {input_path}")
    faces = detections_for_frame(by_frame, frame_index, hold_last)
    out = anonymize_frame(frame, faces, sticker, keep_ids, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    print(f"[완료] 이미지 저장: {output_path}")


def process_video(input_path: Path, output_path: Path, by_frame, sticker,
                  keep_ids: set[str], cfg: StickerConfig,
                  hold_last: bool = False, max_frames: int = 0,
                  video_codec: str = "mp4v") -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {input_path}\n"
            f"  코덱 문제면: ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4 로 재인코딩 후 시도"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*video_codec), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"출력 영상을 만들 수 없습니다: {output_path}")

    idx = 0
    pbar = tqdm(total=total if total > 0 else None, desc="스티커 비식별화")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames > 0 and idx >= max_frames):
                break
            faces = detections_for_frame(by_frame, idx, hold_last)
            writer.write(anonymize_frame(frame, faces, sticker, keep_ids, cfg))
            idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()
    print(f"[완료] {idx} 프레임 처리 -> {output_path} (오디오 미포함)")
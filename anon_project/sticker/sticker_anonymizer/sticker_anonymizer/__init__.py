"""스티커 기반 얼굴 비식별화 패키지 (디퓨전 없음, Colab T4/CPU 친화적)."""

from .compositing import StickerConfig, anonymize_frame
from .detections import detections_for_frame, load_detections
from .pipeline import process_image, process_video
from .stickers import build_sticker

__all__ = [
    "StickerConfig",
    "anonymize_frame",
    "load_detections",
    "detections_for_frame",
    "process_image",
    "process_video",
    "build_sticker",
]
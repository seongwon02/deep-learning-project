"""CLI 진입점.

사용 예:
  python -m sticker_anonymizer \
      --input input.mp4 --detections detections.json --output out.mp4 \
      --emoji 🤖 --keep-track-ids 1,3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .compositing import StickerConfig
from .detections import all_track_ids, load_detections
from .pipeline import input_kind, process_image, process_video
from .stickers import build_sticker


def parse_id_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="스티커 기반 얼굴 비식별화 (디퓨전 없음)")
    p.add_argument("--input", type=Path, required=True, help="입력 이미지/영상")
    p.add_argument("--output", type=Path, required=True, help="출력 경로")
    p.add_argument("--detections", type=Path, required=True, help="YOLO 탐지 JSON")
    p.add_argument("--keep-track-ids", type=parse_id_list, default=set(),
                   help="비식별화에서 제외할 ID(쉼표 구분). 예: 1,5,6")
    p.add_argument("--anonymize-track-ids", type=parse_id_list, default=set(),
                   help="지정 시 이 ID들만 비식별화")

    p.add_argument("--emoji", default="🤖", help="이모지 스티커 (PNG 미지정 시 사용)")
    p.add_argument("--sticker-png", type=Path, help="PNG 스티커 파일(투명 배경 권장). 지정 시 이모지 대신 사용")

    p.add_argument("--box-scale", type=float, default=1.9, help="bbox 한 변 대비 스티커 배율 (이마까지 덮으려면 크게)")
    p.add_argument("--eye-scale", type=float, default=3.2, help="키포인트 있을 때 눈 간격 대비 배율")
    p.add_argument("--y-shift", type=float, default=-0.18,
                   help="스티커 중심을 bbox 높이 대비 세로 이동(이마/머리 덮기). 음수=위로")
    p.add_argument("--min-face-size", type=int, default=40, help="이 픽셀(한 변) 미만 얼굴은 블러")
    p.add_argument("--blur-blocks", type=int, default=8, help="작은 얼굴 픽셀화 블록 수")

    p.add_argument("--frame-index", type=int, default=0, help="이미지 입력 시 사용할 프레임 인덱스")
    p.add_argument("--hold-last-detections", action="store_true",
                   help="JSON에 없는 프레임은 직전 탐지 재사용")
    p.add_argument("--max-frames", type=int, default=0, help="영상 데모용: N프레임에서 중단")
    p.add_argument("--video-codec", default="mp4v")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    by_frame = load_detections(args.detections)
    total_faces = sum(len(v) for v in by_frame.values())
    print(f"[정보] 탐지 프레임 {len(by_frame)}개, 얼굴 레코드 {total_faces}개 로드")

    keep_ids = set(args.keep_track_ids)
    if args.anonymize_track_ids:  # 일부만 비식별화 지정 시, 나머지는 keep으로
        keep_ids |= (all_track_ids(by_frame) - args.anonymize_track_ids)

    sticker = build_sticker(args.sticker_png, args.emoji, size=512)
    cfg = StickerConfig(
        box_scale=args.box_scale,
        eye_scale=args.eye_scale,
        y_shift=args.y_shift,
        min_face_size=args.min_face_size,
        blur_blocks=args.blur_blocks,
    )
    src = f"PNG:{args.sticker_png}" if args.sticker_png else f"emoji:{args.emoji}"
    print(f"[정보] 스티커 소스 = {src}, keep IDs = {sorted(keep_ids) or '없음'}")

    if input_kind(args.input) == "image":
        process_image(args.input, args.output, by_frame, sticker, keep_ids, cfg,
                      frame_index=args.frame_index, hold_last=args.hold_last_detections)
    else:
        process_video(args.input, args.output, by_frame, sticker, keep_ids, cfg,
                      hold_last=args.hold_last_detections, max_frames=args.max_frames,
                      video_codec=args.video_codec)


if __name__ == "__main__":
    main()
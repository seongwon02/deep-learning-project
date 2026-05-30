#!/usr/bin/env python3
"""
영상 -> 얼굴 탐지 JSON 추출 (YOLO, 콘솔 입력 없음 / 코랩 친화적).

yolo11s-pose_widerface.pt 같은 모델로 영상 전체를 트래킹하면서
모든 얼굴의 track_id, bbox(얼굴 외곽)를 프레임별로 뽑아
스티커 단계(sticker_anonymizer)가 바로 읽는 JSON으로 저장한다.

- 얼굴 외곽 bbox만 담는다. keypoints는 담지 않는다.
  스티커는 bbox 영역에 그대로 덮으면 되므로 키포인트가 필요 없다.
- KEEP/제외 선택은 여기서 하지 않는다(모든 얼굴을 다 담는다).
  비식별화 대상 선택은 스티커 단계의 --keep-track-ids 에서 처리.

출력 포맷 (sticker_anonymizer/detections.py가 그대로 파싱):
{
  "frames": [
    {"frame_index": 0,
     "faces": [
       {"track_id": 1, "bbox": [x1,y1,x2,y2], "confidence": 0.97}
     ]}
  ]
}

사용 예 (코랩):
  !python extract_detections.py \
      --input input.mp4 \
      --model yolo11s-pose_widerface.pt \
      --output detections.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def extract(video_path: Path, model_path: Path, output_path: Path,
            conf: float, max_frames: int) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일이 없습니다: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}\n"
            f"  pose 가중치(yolo11s-pose_widerface.pt) 경로를 확인하세요."
        )

    print(f"[정보] 모델 로딩: {model_path}")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}\n"
            f"  코덱 문제면: ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4 로 재인코딩 후 시도"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)
    print(f"[정보] 영상: {width}x{height}, 총 프레임≈{total}")

    frames_out: list[dict] = []
    seen_ids: set[int] = set()
    frame_index = 0
    pbar = tqdm(total=total if total > 0 else None, desc="얼굴 탐지/추적")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames > 0 and frame_index >= max_frames):
                break

            # persist=True: 프레임 간 track_id 유지
            results = model.track(frame, persist=True, conf=conf, verbose=False)
            faces: list[dict] = []

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = (results[0].boxes.conf.cpu().numpy()
                         if results[0].boxes.conf is not None else None)
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    # 트래킹 ID가 없으면 순번으로 임시 부여
                    track_ids = [i + 1 for i in range(len(boxes))]

                # 얼굴 외곽 bbox만 사용한다. pose 모델이라도 keypoints는 담지 않는다.
                # (스티커는 bbox 영역에 그대로 덮는다)
                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face = {
                        "track_id": int(track_id),
                        "bbox": [x1, y1, x2, y2],
                    }
                    if confs is not None and i < len(confs):
                        face["confidence"] = round(float(confs[i]), 4)

                    faces.append(face)
                    seen_ids.add(int(track_id))

            frames_out.append({"frame_index": frame_index, "faces": faces})
            frame_index += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"frames": frames_out}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[완료] {frame_index} 프레임 처리, 고유 ID {len(seen_ids)}개 "
          f"(IDs: {sorted(seen_ids)})")
    print(f"[완료] JSON 저장: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="영상에서 얼굴 탐지 JSON 추출 (YOLO pose)")
    p.add_argument("--input", type=Path, required=True, help="입력 영상")
    p.add_argument("--model", type=Path, required=True,
                   help="YOLO 가중치 (예: yolo11s-pose_widerface.pt)")
    p.add_argument("--output", type=Path, default=Path("detections.json"), help="출력 JSON 경로")
    p.add_argument("--conf", type=float, default=0.3, help="탐지 신뢰도 임계값")
    p.add_argument("--max-frames", type=int, default=0, help="N프레임에서 중단 (0=전체)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(args.input, args.model, args.output, args.conf, args.max_frames)
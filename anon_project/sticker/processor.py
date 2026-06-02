"""
Sticker / Emoji overlay anonymizer processor.
Provides clean Python APIs wrapper over sticker_anonymizer package.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Optional

# Set up paths to properly import the local sticker_anonymizer package
curr_dir = Path(__file__).parent.resolve()
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from sticker_anonymizer.compositing import StickerConfig
from sticker_anonymizer.detections import all_track_ids, load_detections
from sticker_anonymizer.pipeline import input_kind, process_image, process_video
from sticker_anonymizer.stickers import build_sticker

def apply_sticker_anonymization(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    detections: Union[Dict[str, Any], Path, str],
    keep_track_ids: Union[Set[str], List[str], str] = set(),
    anonymize_track_ids: Union[Set[str], List[str], str] = set(),
    emoji_char: str = "🐼",
    sticker_png_path: Optional[Union[str, Path]] = None,
    box_scale: float = 1.9,
    eye_scale: float = 3.2,
    y_shift: float = -0.18,
    min_face_size: int = 40,
    blur_blocks: int = 8,
    frame_index: int = 0,
    hold_last_detections: bool = False,
    max_frames: int = 0,
    video_codec: str = "mp4v"
) -> None:
    """
    Main function to run sticker overlay anonymization.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process detections
    if isinstance(detections, (str, Path)):
        by_frame = load_detections(Path(detections))
    else:
        # Dictionary format
        by_frame = {}
        if isinstance(detections, dict) and "frames" in detections:
            for item in detections["frames"]:
                fidx = item.get("frame_index", 0)
                faces = item.get("faces", [])
                by_frame[fidx] = faces
        else:
            by_frame = detections
            
    # Process keep/anonymize track IDs
    if isinstance(keep_track_ids, str):
        keep_ids = {x.strip() for x in keep_track_ids.split(",") if x.strip()}
    else:
        keep_ids = set(str(x) for x in keep_track_ids)
        
    if isinstance(anonymize_track_ids, str):
        anon_ids = {x.strip() for x in anonymize_track_ids.split(",") if x.strip()}
    else:
        anon_ids = set(str(x) for x in anonymize_track_ids)
        
    if anon_ids:
        # If user specified only certain IDs to anonymize, exclude them from keep_ids
        keep_ids |= (all_track_ids(by_frame) - anon_ids)
        
    # Auto-resolve sticker_png_path if it points to a string name
    if sticker_png_path:
        sticker_png_path = Path(sticker_png_path)
        if not sticker_png_path.exists():
            # Try searching in local assets folder
            local_asset = curr_dir / "assets" / sticker_png_path.name
            if local_asset.exists():
                sticker_png_path = local_asset
                
    sticker = build_sticker(sticker_png_path, emoji_char, size=512)
    cfg = StickerConfig(
        box_scale=box_scale,
        eye_scale=eye_scale,
        y_shift=y_shift,
        min_face_size=min_face_size,
        blur_blocks=blur_blocks,
    )
    
    src = f"PNG:{sticker_png_path}" if sticker_png_path else f"emoji:{emoji_char}"
    print(f"[Sticker Anonymizer] Sticker source = {src}, Keep IDs = {sorted(keep_ids) or 'None'}")
    
    if input_kind(input_path) == "image":
        process_image(
            input_path, output_path, by_frame, sticker, keep_ids, cfg,
            frame_index=frame_index, hold_last=hold_last_detections
        )
    else:
        process_video(
            input_path, output_path, by_frame, sticker, keep_ids, cfg,
            hold_last=hold_last_detections, max_frames=max_frames,
            video_codec=video_codec
        )
    print(f"[Sticker Anonymizer] Processed result saved to {output_path}")

if __name__ == "__main__":
    # Retain CLI ability
    p = argparse.ArgumentParser(description="Sticker Anonymizer CLI wrapper")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--keep-track-ids", type=str, default="")
    p.add_argument("--anonymize-track-ids", type=str, default="")
    p.add_argument("--emoji", default="🐼")
    p.add_argument("--sticker-png", type=Path)
    p.add_argument("--box-scale", type=float, default=1.9)
    p.add_argument("--eye-scale", type=float, default=3.2)
    p.add_argument("--y-shift", type=float, default=-0.18)
    p.add_argument("--min-face-size", type=int, default=40)
    p.add_argument("--blur-blocks", type=int, default=8)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--hold-last-detections", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--video-codec", default="mp4v")
    args = p.parse_args()
    
    apply_sticker_anonymization(
        args.input, args.output, args.detections,
        keep_track_ids=args.keep_track_ids,
        anonymize_track_ids=args.anonymize_track_ids,
        emoji_char=args.emoji,
        sticker_png_path=args.sticker_png,
        box_scale=args.box_scale,
        eye_scale=args.eye_scale,
        y_shift=args.y_shift,
        min_face_size=args.min_face_size,
        blur_blocks=args.blur_blocks,
        frame_index=args.frame_index,
        hold_last_detections=args.hold_last_detections,
        max_frames=args.max_frames,
        video_codec=args.video_codec
    )

#!/usr/bin/env python3
"""Build SDXL prompt text from non-identifying reference image traits."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_NEGATIVE_PROMPT = (
    "same identity as reference, celebrity, famous person, exact facial match, "
    "watermark, text, distorted face, uncanny"
)
STYLE_SUFFIX = (
    "synthetic non-famous person, realistic natural face, consistent facial anatomy, "
    "matching lighting and camera style"
)


def parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def image_feature_record(path: Path) -> dict[str, Any]:
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    saturation = float(hsv[..., 1].mean() / 255.0)
    red_mean = float(rgb[..., 0].mean())
    blue_mean = float(rgb[..., 2].mean())
    warmth = (red_mean - blue_mean) / 255.0

    tags = []
    if brightness < 0.32:
        tags.append("low-key lighting")
    elif brightness > 0.68:
        tags.append("bright soft lighting")
    else:
        tags.append("balanced natural lighting")

    if contrast < 0.16:
        tags.append("soft low-contrast image")
    elif contrast > 0.30:
        tags.append("defined facial contrast")

    if saturation < 0.18:
        tags.append("muted color palette")
    elif saturation > 0.45:
        tags.append("vivid color palette")

    if warmth > 0.06:
        tags.append("warm color temperature")
    elif warmth < -0.06:
        tags.append("cool color temperature")
    else:
        tags.append("neutral color temperature")

    if image.width >= image.height * 1.25:
        tags.append("landscape framing")
    elif image.height >= image.width * 1.25:
        tags.append("portrait framing")
    else:
        tags.append("square portrait framing")

    return {
        "path": str(path),
        "width": image.width,
        "height": image.height,
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "saturation": round(saturation, 4),
        "warmth": round(warmth, 4),
        "tags": tags,
    }


def captioner_device(device: str) -> int:
    if device == "cuda":
        return 0
    if device == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    return -1


def load_captioner(model_name: str, device: str) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Caption mode requires transformers.") from exc
    return pipeline("image-to-text", model=model_name, device=captioner_device(device))


def caption_image(captioner: Any, path: Path) -> str | None:
    image = Image.open(path).convert("RGB")
    result = captioner(image)
    if not result:
        return None
    first = result[0]
    text = first.get("generated_text") if isinstance(first, dict) else str(first)
    return sanitize_caption(text) if text else None


def sanitize_caption(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(a|an|the) photo of ", "", text)
    text = text.replace("a picture of ", "")
    text = text.replace("the same person", "a synthetic non-famous person")
    text = text.replace("same person", "synthetic non-famous person")
    text = text.replace("celebrity", "non-famous person")
    return text.strip(" .,")


def common_tags(feature_records: list[dict[str, Any]]) -> list[str]:
    if not feature_records:
        return []
    tag_counts: Counter[str] = Counter()
    for record in feature_records:
        tag_counts.update(record["tags"])
    threshold = max(1, int(np.ceil(len(feature_records) * 0.5)))
    return [tag for tag, count in tag_counts.most_common() if count >= threshold]


def concise_caption_phrases(captions: list[str], limit: int = 3) -> list[str]:
    seen = set()
    phrases = []
    for caption in captions:
        cleaned = sanitize_caption(caption)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        phrases.append(cleaned)
        if len(phrases) >= limit:
            break
    return phrases


def build_prompt_text(
    feature_records: list[dict[str, Any]],
    captions: list[str],
    manual_tags: list[str],
) -> str:
    parts = [STYLE_SUFFIX]
    parts.extend(concise_caption_phrases(captions))
    parts.extend(common_tags(feature_records))
    parts.extend(manual_tags)

    deduped = []
    seen = set()
    for part in parts:
        normalized = part.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(part.strip())
    return ", ".join(deduped)


def build_reference_prompt(
    image_paths: list[Path],
    mode: str = "heuristic",
    caption_model: str = DEFAULT_CAPTION_MODEL,
    device: str = "auto",
    manual_tags: list[str] | None = None,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("At least one reference image is required.")

    manual_tags = manual_tags or []
    feature_records = [image_feature_record(path) for path in image_paths]
    captions: list[str] = []
    warnings: list[str] = []

    if mode in {"caption", "auto"}:
        try:
            captioner = load_captioner(caption_model, device)
            for path in image_paths:
                caption = caption_image(captioner, path)
                if caption:
                    captions.append(caption)
        except Exception as exc:
            if mode == "caption":
                raise
            warnings.append(f"caption_fallback:{type(exc).__name__}")

    prompt = build_prompt_text(
        feature_records=feature_records,
        captions=captions,
        manual_tags=manual_tags,
    )
    return {
        "reference_count": len(image_paths),
        "mode": mode,
        "caption_model": caption_model if captions else None,
        "prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "captions": captions,
        "shared_tags": common_tags(feature_records),
        "manual_tags": manual_tags,
        "features": feature_records,
        "warnings": warnings,
    }


def build_reference_prompt_from_paths(
    image_paths: list[Path],
    mode: str = "heuristic",
    caption_model: str = DEFAULT_CAPTION_MODEL,
    device: str = "auto",
    manual_tags: list[str] | None = None,
) -> dict[str, Any]:
    return build_reference_prompt(
        image_paths=[Path(path) for path in image_paths],
        mode=mode,
        caption_model=caption_model,
        device=device,
        manual_tags=manual_tags,
    )


def load_reference_prompt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("prompt"):
        raise ValueError("Reference prompt JSON must contain a 'prompt' field.")
    return payload


def write_reference_prompt(payload: dict[str, Any], output_json: Path | None, output_text: Path | None) -> None:
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_text:
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text(str(payload["prompt"]) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract non-identifying reference image traits into an SDXL prompt.",
    )
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-text", type=Path)
    parser.add_argument("--mode", choices=("heuristic", "caption", "auto"), default="heuristic")
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--manual-tags",
        type=parse_csv,
        default=[],
        help="Comma-separated human-provided traits to append, e.g. 'round glasses, short black hair'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_reference_prompt_from_paths(
        image_paths=args.images,
        mode=args.mode,
        caption_model=args.caption_model,
        device=args.device,
        manual_tags=args.manual_tags,
    )
    write_reference_prompt(payload, args.output_json, args.output_text)
    print(payload["prompt"])


if __name__ == "__main__":
    main()

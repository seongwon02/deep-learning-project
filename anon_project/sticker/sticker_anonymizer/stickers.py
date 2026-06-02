"""스티커 소스: 컬러 이모지 렌더링 또는 PNG 로드.

PNG 경로가 주어지면 PNG를 우선 사용하고, 없으면 이모지를 렌더링한다.
코랩 기본 Noto Color Emoji는 109px 비트맵 strike만 지원하므로
해당 크기로 렌더한 뒤 원하는 크기로 리사이즈한다.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EMOJI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS 로컬 테스트용
    "C:/Windows/Fonts/seguiemj.ttf",                # Windows 로컬 테스트용
]
NOTO_EMOJI_STRIKE = 109


def _find_emoji_font() -> str | None:
    for p in EMOJI_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _to_square(img: Image.Image, size: int) -> Image.Image:
    side = max(img.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2), img)
    return square.resize((size, size), Image.LANCZOS)


def render_emoji_sticker(emoji_char: str, size: int = 512) -> Image.Image:
    font_path = _find_emoji_font()
    if font_path is None:
        raise RuntimeError(
            "컬러 이모지 폰트를 찾지 못했습니다. 코랩이면 다음을 실행하세요:\n"
            "  !apt-get -y install fonts-noto-color-emoji\n"
            "또는 --sticker-png 로 PNG 스티커를 지정하세요."
        )

    strike = NOTO_EMOJI_STRIKE if "Noto" in font_path else size
    try:
        font = ImageFont.truetype(font_path, strike)
    except OSError:
        font = ImageFont.truetype(font_path, size)

    canvas = Image.new("RGBA", (strike * 2, strike * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        draw.text((strike // 2, strike // 2), emoji_char, font=font, embedded_color=True)
    except TypeError:
        draw.text((strike // 2, strike // 2), emoji_char, font=font)

    bbox = canvas.getbbox()
    if bbox is None:
        raise RuntimeError(f"이모지 '{emoji_char}' 렌더 결과가 비었습니다.")
    return _to_square(canvas.crop(bbox), size)


def load_png_sticker(path: Path, size: int = 512) -> Image.Image:
    return _to_square(Image.open(path).convert("RGBA"), size)


def build_sticker(sticker_png: Path | None, emoji: str, size: int = 512) -> Image.Image:
    if sticker_png:
        return load_png_sticker(Path(sticker_png), size=size)
    return render_emoji_sticker(emoji, size=size)
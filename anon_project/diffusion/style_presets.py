#!/usr/bin/env python3
"""Style presets for clip-level face anonymization.

Each preset replaces the detected face region with a stylised object
(helmet, animal mask, robot mask, hood) that completely occludes the
original identity.  All prompts are designed so that *no* human facial
features remain visible in the generated output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields

# ---------------------------------------------------------------------------
# Defaults mirroring anonymize.py so we can detect user overrides
# ---------------------------------------------------------------------------

_ARGPARSE_DEFAULTS: dict[str, object] = {
    "prompt": (
        "photorealistic face of a synthetic non-famous person, natural skin texture, "
        "realistic eyes, matching head pose, matching lighting, high detail"
    ),
    "negative_prompt": (
        "celebrity, famous person, same identity, cartoon, anime, 3d render, doll, "
        "plastic skin, deformed face, asymmetrical eyes, bad anatomy, mask artifact, "
        "uncanny, blurry"
    ),
    "controlnet": "none",
    "controlnet_scale": 0.55,
    "ip_adapter_scale": 0.75,
    "strength": 0.98,
    "guidance_scale": 5.0,
    "num_inference_steps": 28,
    "mask_expansion": 1.35,
    "mask_dilation": 9,
}

# ---------------------------------------------------------------------------
# StylePreset dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StylePreset:
    """A complete anonymization style that replaces a face with an object.

    Attributes:
        name: Short machine-readable identifier (e.g. ``"helmet"``).
        prompt: Positive diffusion prompt describing the replacement.
        negative_prompt: Negative prompt steering generation away from
            undesired artefacts.
        privacy_negative_prompt: Extra negative prompt tokens that
            specifically forbid any recognisable human facial features.
        controlnet: Recommended ControlNet conditioning mode
            (``"canny"``, ``"depth"``, or ``"none"``).
        controlnet_scale: ControlNet conditioning scale.
        ip_adapter_scale: IP-Adapter image influence scale.
        strength: Inpainting denoising strength (0.0–1.0).
        guidance_scale: Classifier-free guidance scale.
        num_inference_steps: Number of diffusion sampling steps.
        mask_expansion: Multiplicative expansion applied to the face
            bounding box before building the inpainting mask.
        mask_dilation: Dilation kernel size (px) applied to the binary
            mask before blurring.
    """

    name: str
    prompt: str
    negative_prompt: str
    privacy_negative_prompt: str
    controlnet: str  # "canny" | "depth" | "none"
    controlnet_scale: float
    ip_adapter_scale: float
    strength: float
    guidance_scale: float
    num_inference_steps: int
    mask_expansion: float
    mask_dilation: int


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

STYLE_PRESETS: dict[str, StylePreset] = {
    # ---- helmet ----
    "helmet": StylePreset(
        name="helmet",
        prompt=(
            "futuristic motorcycle helmet with opaque black visor covering the entire "
            "face and head, no face visible, no facial features visible, full-face "
            "occlusion, reflective visor surface, sleek aerodynamic shell, matte black "
            "and carbon fiber finish, sci-fi helmet, photorealistic product render, "
            "matching head pose, matching scene lighting, high detail"
        ),
        negative_prompt=(
            "cracked visor, transparent visor, see-through, broken helmet, "
            "open visor, removed helmet, cartoon, anime, 3d render, doll, "
            "plastic skin, bad anatomy, blurry, low quality, watermark, text"
        ),
        privacy_negative_prompt=(
            "realistic human face, human skin, human eyes, human nose, human mouth, "
            "human lips, human teeth, visible face, facial features, skin texture, "
            "face identity, recognizable person, celebrity likeness, real person, "
            "face swap, deepfake, portrait photography"
        ),
        controlnet="canny",
        controlnet_scale=0.45,
        ip_adapter_scale=0.3,
        strength=0.99,
        guidance_scale=7.0,
        num_inference_steps=35,
        mask_expansion=1.55,
        mask_dilation=12,
    ),
    # ---- animal_mask ----
    "animal_mask": StylePreset(
        name="animal_mask",
        prompt=(
            "cute oversized panda bear head mask covering the entire face and head, "
            "no face visible, no facial features visible, full-face occlusion, "
            "plush fur texture, round black and white panda head, button eyes, "
            "soft fabric animal mask, opaque solid mask surface, "
            "matching head pose, matching scene lighting, photorealistic, high detail"
        ),
        negative_prompt=(
            "transparent mask, mesh mask, see-through, half mask, eye holes showing "
            "real eyes, cartoon, anime, 3d render, doll, plastic skin, bad anatomy, "
            "blurry, low quality, watermark, text, uncanny valley"
        ),
        privacy_negative_prompt=(
            "realistic human face, human skin, human eyes, human nose, human mouth, "
            "human lips, human teeth, visible face, facial features, skin texture, "
            "face identity, recognizable person, celebrity likeness, real person, "
            "face swap, deepfake, portrait photography"
        ),
        controlnet="canny",
        controlnet_scale=0.40,
        ip_adapter_scale=0.25,
        strength=0.99,
        guidance_scale=7.5,
        num_inference_steps=35,
        mask_expansion=1.60,
        mask_dilation=14,
    ),
    # ---- robot_mask ----
    "robot_mask": StylePreset(
        name="robot_mask",
        prompt=(
            "metallic chrome robot head replacing the entire face and head, "
            "no face visible, no facial features visible, full-face occlusion, "
            "glowing LED eyes, smooth brushed-metal faceplate, mechanical jaw panel, "
            "futuristic android head, opaque solid metal surface, "
            "matching head pose, matching scene lighting, photorealistic, high detail"
        ),
        negative_prompt=(
            "human skin showing through, transparent panels, see-through, organic face, "
            "cartoon, anime, 3d render, doll, plastic skin, bad anatomy, "
            "blurry, low quality, watermark, text, cheap toy"
        ),
        privacy_negative_prompt=(
            "realistic human face, human skin, human eyes, human nose, human mouth, "
            "human lips, human teeth, visible face, facial features, skin texture, "
            "face identity, recognizable person, celebrity likeness, real person, "
            "face swap, deepfake, portrait photography"
        ),
        controlnet="depth",
        controlnet_scale=0.50,
        ip_adapter_scale=0.30,
        strength=0.99,
        guidance_scale=7.0,
        num_inference_steps=35,
        mask_expansion=1.55,
        mask_dilation=12,
    ),
    # ---- hood ----
    "hood": StylePreset(
        name="hood",
        prompt=(
            "dark oversized hood pulled deep over the head with industrial goggles "
            "covering the entire face, no face visible, no facial features visible, "
            "full-face occlusion, opaque tinted goggle lenses, heavy fabric shadow "
            "inside the hood hiding all skin, mysterious hooded figure, "
            "matching head pose, matching scene lighting, photorealistic, high detail"
        ),
        negative_prompt=(
            "visible face inside hood, exposed skin, transparent goggles showing eyes, "
            "hood pulled back, cartoon, anime, 3d render, doll, plastic skin, "
            "bad anatomy, blurry, low quality, watermark, text"
        ),
        privacy_negative_prompt=(
            "realistic human face, human skin, human eyes, human nose, human mouth, "
            "human lips, human teeth, visible face, facial features, skin texture, "
            "face identity, recognizable person, celebrity likeness, real person, "
            "face swap, deepfake, portrait photography"
        ),
        controlnet="canny",
        controlnet_scale=0.45,
        ip_adapter_scale=0.25,
        strength=0.99,
        guidance_scale=6.5,
        num_inference_steps=32,
        mask_expansion=1.65,
        mask_dilation=14,
    ),
    # ---- cardboard_box ----
    "cardboard_box": StylePreset(
        name="cardboard_box",
        prompt=(
            "a simple plain brown square cardboard box covering the entire head, "
            "no face visible, no facial features visible, full-head occlusion, "
            "flat paperboard surfaces, clean straight edges, simple folded box corners, "
            "danbo style cardboard head, matching head pose, matching scene lighting, "
            "photorealistic, minimal design"
        ),
        negative_prompt=(
            "realistic face, eyes, nose, mouth, visor, helmet, organic skin, "
            "detailed robot, complex helmet design, curves, cartoon, anime, doll, "
            "blurry, low quality, watermark, text, logos"
        ),
        privacy_negative_prompt=(
            "realistic human face, human skin, human eyes, human nose, human mouth, "
            "human lips, human teeth, visible face, facial features, skin texture, "
            "face identity, recognizable person, celebrity likeness, real person, "
            "face swap, deepfake, portrait photography"
        ),
        controlnet="depth",
        controlnet_scale=0.55,
        ip_adapter_scale=0.15,
        strength=0.99,
        guidance_scale=6.5,
        num_inference_steps=30,
        mask_expansion=1.65,
        mask_dilation=14,
     ),
    # ---- 2d_animation ----
    "2d_animation": StylePreset(
        name="2d_animation",
        prompt=(
            "beautiful 2D anime style digital painting, masterpiece, highly detailed face, "
            "vibrant colors, clean lineart, anime aesthetic, stylized illustration, "
            "matching head pose, matching scene lighting, high detail"
        ),
        negative_prompt=(
            "photorealistic, 3d render, photograph, real skin texture, bad anatomy, "
            "deformed face, blurry, low quality, watermark, text"
        ),
        privacy_negative_prompt=(
            "realistic human face, human skin, human eyes, human nose, human mouth, "
            "human lips, human teeth, visible face, facial features, skin texture, "
            "face identity, recognizable person, celebrity likeness, real person, "
            "face swap, deepfake, portrait photography"
        ),
        controlnet="none",
        controlnet_scale=0.55,
        ip_adapter_scale=0.75,
        strength=0.98,
        guidance_scale=5.0,
        num_inference_steps=28,
        mask_expansion=1.35,
        mask_dilation=9,
    ),
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_preset(name: str) -> StylePreset:
    """Return the :class:`StylePreset` matching *name*.

    Parameters:
        name: Case-insensitive preset key (e.g. ``"helmet"``).

    Returns:
        The corresponding :class:`StylePreset` instance.

    Raises:
        ValueError: If *name* does not match any known preset.
    """
    key = name.strip().lower()
    if key not in STYLE_PRESETS:
        available = ", ".join(sorted(STYLE_PRESETS))
        raise ValueError(
            f"Unknown style preset {name!r}. "
            f"Available presets: {available}"
        )
    return STYLE_PRESETS[key]


def list_presets() -> list[str]:
    """Return a sorted list of available preset names.

    Returns:
        List of preset name strings, e.g.
        ``['animal_mask', 'helmet', 'hood', 'robot_mask']``.
    """
    return sorted(STYLE_PRESETS)


def apply_preset(args: argparse.Namespace, preset: StylePreset) -> None:
    """Apply *preset* settings to an :class:`~argparse.Namespace` **in-place**.

    Only fields that the user has **not** explicitly overridden (i.e. those
    that still hold their argparse default value) are updated.  This lets a
    user do ``--strength 0.85 --style helmet`` and keep their custom strength
    while still receiving the helmet prompt, controlnet mode, etc.

    The ``privacy_negative_prompt`` from the preset is **always** appended to
    ``args.negative_prompt`` so that facial-feature suppression is guaranteed
    regardless of the user's custom negative prompt.

    Parameters:
        args: Parsed CLI namespace — modified in-place.
        preset: The :class:`StylePreset` to apply.
    """
    # Map StylePreset field names → argparse attribute names.
    # Most are identical; any future rename can be handled here.
    _field_to_attr: dict[str, str] = {
        "prompt": "prompt",
        "negative_prompt": "negative_prompt",
        "controlnet": "controlnet",
        "controlnet_scale": "controlnet_scale",
        "ip_adapter_scale": "ip_adapter_scale",
        "strength": "strength",
        "guidance_scale": "guidance_scale",
        "num_inference_steps": "num_inference_steps",
        "mask_expansion": "mask_expansion",
        "mask_dilation": "mask_dilation",
    }

    for field_name, attr_name in _field_to_attr.items():
        preset_value = getattr(preset, field_name)
        current_value = getattr(args, attr_name, None)
        default_value = _ARGPARSE_DEFAULTS.get(attr_name)

        # Override only when the current value equals the argparse default,
        # meaning the user did not explicitly set it on the command line.
        if default_value is not None and current_value == default_value:
            setattr(args, attr_name, preset_value)

    # Always append the privacy negative prompt so face features are
    # actively suppressed even when the user supplies a custom negative
    # prompt.
    current_neg = getattr(args, "negative_prompt", "")
    privacy_neg = preset.privacy_negative_prompt
    if privacy_neg and privacy_neg not in current_neg:
        separator = ", " if current_neg else ""
        setattr(args, "negative_prompt", current_neg + separator + privacy_neg)

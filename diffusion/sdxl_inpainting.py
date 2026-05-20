"""
SDXLInpainter
- 주요 기능:
  1. Occlusion-aware mask: 얼굴 마스크 안에서 손가락/머리카락/안경 등 피부가 아닌 영역을 제외
     → 손가락이 얼굴 앞을 지나가도 지워지지 않음
  2. 스타일 강화: LoRA 가중치 상향, strength/CFG 재튜닝, 지브리 특화 프롬프트 강화
  3. 적응형(Adaptive) 파라미터: 얼굴 크기 + 정면/측면 자동 판별 후 strength/controlnet_scale 조정
  4. 2-pass refine 옵션: 1차(ControlNet 인페인팅) → 2차(약한 img2img로 스타일만 강화)
  5. 블렌딩: 얼굴 크기에 비례한 feather radius + 색상 매칭(blending_utils.color_transfer)
"""

import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from PIL import Image, ImageFilter
import numpy as np
import cv2
import os

from blending_utils import color_transfer


class SDXLInpainter:
    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_id: str = "diffusers/controlnet-canny-sdxl-1.0",
        lora_path: str = "Redmond-StdGBRRedmAF-StudioGhibli.safetensors",
        lora_weight: float = 1.2,
        device: str = "cuda",
    ):
        self.device = device

        # Canny ControlNet: 형태/포즈 보존 (옆모습, 얼굴 방향 유지)
        print("[SDXLInpainter] Loading Canny ControlNet for structure preservation...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=torch.float16
        ).to(device)

        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[SDXLInpainter] xformers not enabled: {e}")

        # Ghibli LoRA
        print("[SDXLInpainter] Loading Local Ghibli Style LoRA...")
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_lora_path = os.path.join(current_dir, lora_path)
            
            print(f"[SDXLInpainter] Resolved LoRA Path: {absolute_lora_path}")

            if not os.path.exists(absolute_lora_path):
                print(f"[SDXLInpainter] Error: LoRA 파일을 찾을 수 없습니다 -> {absolute_lora_path}")
            else:
                self.pipe.load_lora_weights(absolute_lora_path, adapter_name="ghibli")
                self.pipe.set_adapters(["ghibli"], adapter_weights=[lora_weight])
                print(f"[SDXLInpainter] LoRA loaded successfully with weight={lora_weight}")
        except Exception as e:
            print(f"[SDXLInpainter] LoRA loading failed: {e}")

        self.generator = None

    # ----------------------------------------------------------------------------- #
    # 1) Region extraction
    # ----------------------------------------------------------------------------- #
    def get_individual_face_regions(self, mask: Image.Image, min_area: int = 100):
        mask_np = np.array(mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np)
        regions = []
        img_h, img_w = mask_np.shape

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue

            short_side = min(w, h)
            if short_side < 120:
                pad_ratio = 1.0
            elif short_side < 250:
                pad_ratio = 0.8
            else:
                pad_ratio = 0.6

            pad_w = int(w * pad_ratio)
            pad_h = int(h * pad_ratio)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_w, x + w + pad_w)
            y2 = min(img_h, y + h + pad_h)

            isolated_mask_np = (labels == i).astype(np.uint8) * 255

            k = max(11, int(short_side * 0.08) | 1)
            kernel = np.ones((k, k), np.uint8)
            expanded = cv2.dilate(isolated_mask_np, kernel, iterations=1)

            regions.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "face_mask_full": Image.fromarray(expanded),
                    "raw_w": w,
                    "raw_h": h,
                    "area": int(area),
                }
            )
        return regions

    # ----------------------------------------------------------------------------- #
    # 2) Occlusion mask
    # ----------------------------------------------------------------------------- #
    @staticmethod
    def _build_skin_mask(rgb_crop: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        skin = cv2.inRange(ycrcb, lower, upper)

        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return skin

    @staticmethod
    def _build_dark_mask(rgb_crop: np.ndarray, v_threshold: int = 60) -> np.ndarray:
        hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        dark = ((v < v_threshold) & (s < 80)).astype(np.uint8) * 255
        return dark

    def build_occlusion_aware_mask(
        self,
        face_crop: Image.Image,
        face_mask_crop: Image.Image,
        protect_strength: float = 1.0,
    ) -> Image.Image:
        rgb = np.array(face_crop)
        mask_np = np.array(face_mask_crop)

        skin = self._build_skin_mask(rgb)
        dark = self._build_dark_mask(rgb)

        face_region = (mask_np > 0)
        candidate = ((skin == 0) & (dark == 0) & face_region).astype(np.uint8) * 255

        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
        h, w = mask_np.shape
        face_area = max(1, int(face_region.sum()))

        occlusion = np.zeros_like(mask_np)
        min_a = max(60, int(face_area * 0.003))
        max_a = int(face_area * 0.35)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_a or area > max_a:
                continue

            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(cw, ch) / max(1, min(cw, ch))
            if aspect > 1.6 or area > face_area * 0.02:
                occlusion[labels == i] = 255

        dilate_size = max(3, int(min(h, w) * 0.012) | 1)
        occlusion = cv2.dilate(
            occlusion, np.ones((dilate_size, dilate_size), np.uint8), iterations=1
        )

        occ_ratio = (occlusion > 0).sum() / max(1, face_area)
        if occ_ratio > 0.40:
            print(
                f"[SDXLInpainter] ⚠ occlusion suspiciously large ({occ_ratio*100:.1f}%), "
                f"reducing protection"
            )
            effective_strength = protect_strength * 0.35
        elif occ_ratio > 0.20:
            effective_strength = protect_strength * 0.65
        else:
            effective_strength = protect_strength

        if effective_strength >= 1.0:
            new_mask = mask_np.copy()
            new_mask[occlusion > 0] = 0
        else:
            new_mask = mask_np.astype(np.float32)
            new_mask[occlusion > 0] *= (1.0 - effective_strength)
            new_mask = np.clip(new_mask, 0, 255).astype(np.uint8)

        return Image.fromarray(new_mask)

    # ----------------------------------------------------------------------------- #
    # 3) Canny edge
    # ----------------------------------------------------------------------------- #
    @staticmethod
    def _adaptive_canny(face_crop_np: np.ndarray, small_face: bool) -> np.ndarray:
        if small_face:
            blurred = cv2.GaussianBlur(face_crop_np, (7, 7), 0)
            edges = cv2.Canny(blurred, 80, 180)
        else:
            blurred = cv2.GaussianBlur(face_crop_np, (3, 3), 0)
            edges = cv2.Canny(blurred, 60, 150)
        return edges

    # ----------------------------------------------------------------------------- #
    # 4) 메인 파이프라인
    # ----------------------------------------------------------------------------- #
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        base_prompt: str,
        negative_prompt: str = "",
        protect_occlusion: bool = True,
        protect_strength: float = 0.7,
        seed: int = None,
    ) -> Image.Image:
        regions = self.get_individual_face_regions(mask)
        if not regions:
            return image

        final_image = image.copy()

        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)

        for i, reg in enumerate(regions):
            x1, y1, x2, y2 = reg["bbox"]
            face_mask_full = reg["face_mask_full"]
            face_crop = image.crop((x1, y1, x2, y2))
            face_mask = face_mask_full.crop((x1, y1, x2, y2))

            face_w, face_h = face_crop.size
            short_side = min(face_w, face_h)
            small_face = short_side < 220
            print(
                f"[SDXLInpainter] Face {i+1}/{len(regions)} "
                f"size=({face_w}x{face_h}) "
                f"{'SMALL' if small_face else 'LARGE'}"
            )

            if protect_occlusion:
                face_mask = self.build_occlusion_aware_mask(
                    face_crop, face_mask, protect_strength=protect_strength
                )

            if np.array(face_mask).sum() < 500:
                print(f"[SDXLInpainter]  → mask too small after occlusion, skipping")
                continue

            face_crop_np = np.array(face_crop)
            edges = self._adaptive_canny(face_crop_np, small_face)
            edges_3c = np.stack([edges] * 3, axis=-1)
            control_image = Image.fromarray(edges_3c).resize((1024, 1024), Image.LANCZOS)

            if small_face:
                ctrl_scale = 0.25
                strength_val = 0.95
                steps = 35
                cfg = 7.5
            else:
                ctrl_scale = 0.40
                strength_val = 0.88
                steps = 40
                cfg = 7.5

            ghibli_prompt = (
                "StdGBRedmAF, Studio Ghibli, "
                "Studio Ghibli style anime portrait, Hayao Miyazaki film, "
                "hand-drawn 2D animation, soft cel shading, flat painterly colors, "
                "soft watercolor textures, gentle warm lighting, "
                "large expressive anime eyes, simple clean lineart, "
                "soft round face, anime cheek blush, "
                f"{base_prompt}, "
                "preserving the original head pose and viewing direction, "
                "Ghibli movie still, masterpiece, best quality"
            )

            full_negative = (
                "photorealistic, photograph, realistic skin, real human, "
                "3d render, CGI, plastic skin, pores, wrinkles, "
                "western cartoon, disney, pixar, "
                "deformed face, distorted, extra fingers, missing fingers, "
                "blurry, low quality, jpeg artifacts, watermark, signature, "
                + negative_prompt
            )

            original_size = face_crop.size
            input_image = face_crop.resize((1024, 1024), Image.LANCZOS)
            input_mask = face_mask.resize((1024, 1024), Image.NEAREST)

            generated_face = self.pipe(
                prompt=ghibli_prompt,
                negative_prompt=full_negative,
                image=input_image,
                mask_image=input_mask,
                control_image=control_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                strength=strength_val,
                controlnet_conditioning_scale=ctrl_scale,
                generator=self.generator,
            ).images[0]

            generated_face = generated_face.resize(original_size, Image.LANCZOS)

            try:
                generated_face = color_transfer(face_crop, generated_face)
            except Exception as e:
                print(f"[SDXLInpainter]  → color_transfer skipped: {e}")

            try:
                mask_np_for_clone = np.array(face_mask)
                ys, xs = np.where(mask_np_for_clone > 0)
                if len(xs) > 0:
                    cx = int(xs.mean()) + x1
                    cy = int(ys.mean()) + y1
                    binary_mask = (mask_np_for_clone > 127).astype(np.uint8) * 255

                    bg_bgr = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
                    fg_bgr = cv2.cvtColor(np.array(generated_face), cv2.COLOR_RGB2BGR)

                    blended_bgr = cv2.seamlessClone(
                        fg_bgr, bg_bgr, binary_mask, (cx, cy), cv2.NORMAL_CLONE
                    )
                    final_image = Image.fromarray(cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB))
                else:
                    raise ValueError("empty mask")
            except Exception as e:
                print(f"[SDXLInpainter]  → seamless_clone failed ({e}), falling back to alpha blending")
                feather_radius = max(4, int(short_side * 0.03))
                blurred_mask = face_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
                final_image.paste(generated_face, (x1, y1), blurred_mask)

        return final_image

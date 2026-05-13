# FILE: deep-learning-project/diffusion/sdxl_inpainting_v5.py

import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from PIL import Image, ImageFilter
import numpy as np
import cv2

class SDXLInpainterV5:
    def __init__(self, model_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device="cuda"):
        self.device = device
        
        # [Surgical Modification 1] 형태 보존과 손가락 방어를 위한 Canny ControlNet 로드
        print("Loading Canny ControlNet for Structure Preservation...")
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", 
            torch_dtype=torch.float16
        ).to(device)

        # [Surgical Modification 2] ControlNet이 결합된 인페인팅 파이프라인으로 교체
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()

        print("Loading Ghibli Style LoRA...")
        try:
            self.pipe.load_lora_weights("ProomptEngineer/pe-ghibli-style-sdxl", weight_name="pe_ghibli_style_v1.safetensors", adapter_name="ghibli")
            self.pipe.set_adapters(["ghibli"], adapter_weights=[0.85])
            print("LoRA loaded successfully.")
        except Exception as e:
            print(f"LoRA loading failed: {e}")

    def get_individual_face_regions(self, mask, padding_ratio=0.6):
        mask_np = np.array(mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
        regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 100: continue
            
            pad_w = int(w * 0.8) 
            pad_h = int(h * 0.8)
            img_h, img_w = mask_np.shape
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_w, x + w + pad_w)
            y2 = min(img_h, y + h + pad_h)
            
            isolated_mask_np = (labels == i).astype(np.uint8) * 255
            
            # 마스크는 이전처럼 넉넉하게 확장 (머리카락 전체 커버용)
            kernel = np.ones((50, 50), np.uint8) 
            expanded_mask_np = cv2.dilate(isolated_mask_np, kernel, iterations=1)
            isolated_mask = Image.fromarray(expanded_mask_np)
            
            regions.append((x1, y1, x2, y2, isolated_mask))
        return regions

    def inpaint_v5(self, image, mask, base_prompt, negative_prompt=""):
        regions = self.get_individual_face_regions(mask)
        if not regions: return image

        final_image = image.copy()
        
        for i, (x1, y1, x2, y2, isolated_mask) in enumerate(regions):
            print(f"Processing face {i+1}/{len(regions)} with ControlNet & Alpha Feathering...")
            
            face_crop = image.crop((x1, y1, x2, y2))
            face_mask = isolated_mask.crop((x1, y1, x2, y2))
            
            # [Surgical Modification 3] 원본 얼굴에서 Canny(윤곽선) 추출
            face_crop_np = np.array(face_crop)
            edges = cv2.Canny(face_crop_np, 100, 200) #
            edges_3c = np.stack([edges]*3, axis=-1)
            control_image = Image.fromarray(edges_3c).resize((1024, 1024), Image.NEAREST)
            
            original_size = face_crop.size
            input_image = face_crop.resize((1024, 1024), Image.LANCZOS)
            input_mask = face_mask.resize((1024, 1024), Image.NEAREST)

            # 프롬프트는 2D 작화 퀄리티 유지
            full_prompt = f"{base_prompt}, Studio Ghibli style, detailed anime character, detailed hair, beautiful cinematic lighting, masterpiece"
            
            generated_face = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                mask_image=input_mask,
                control_image=control_image,  # 추출한 손가락/얼굴 윤곽선 투입
                num_inference_steps=40,
                guidance_scale=9.0,
                strength=0.95,                # 작화(화풍)는 완전히 지브리로 덮어씌움
                controlnet_conditioning_scale=0.5 # 원본 형태를 50% 정도 반영 (너무 높으면 실사처럼 선이 지저분해짐)
            ).images[0]

            generated_face = generated_face.resize(original_size, Image.LANCZOS)

            # Alpha Feathering 방식으로 부드럽게 원본에 안착
            blurred_mask = face_mask.filter(ImageFilter.GaussianBlur(radius=20))
            final_image.paste(generated_face, (x1, y1), blurred_mask)
            
        return final_image
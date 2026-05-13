import os
import json
import torch
from PIL import Image
from sdxl_inpainting import SDXLInpainterV5
from video_utils import json_to_mask

def run_v5_pipeline(image_path, json_path, prompt, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        face_data = json.load(f)
    
    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    mask_image = json_to_mask((height, width, 3), face_data)
    
    print("Initializing SDXL Inpainter V5 (Seamless Blending & Color Transfer)...")
    inpainter = SDXLInpainterV5()
    
    refined_prompt = f"masterpiece, best quality, ghibli style, {prompt} style anime character face, cel shaded, flat colors, distinctive expressive anime eyes, beautifully drawn, 2D illustration"
    negative_prompt = "realistic, 3d, photorealistic, photograph, CGI, human skin, human anatomy, deformed, distorted, messy, blurry, low quality, bad anatomy"
    
    print(f"Running v5 pipeline...")
    result_image = inpainter.inpaint_v5(
        image=original_image,
        mask=mask_image,
        base_prompt=refined_prompt,
        negative_prompt=negative_prompt
    )
    
    result_image.save(output_path)
    print(f"V5 Seamless result saved to: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    img_path = os.path.join(project_root, "yolo_blur", "dataset", "test.jpg")
    json_path = os.path.join(project_root, "yolo_blur", "yolo_result.json")
        
    if not os.path.exists(json_path):
        print(f"yolo_result.json not found.")
    else:
        output_path = os.path.join(base_dir, "final_character_result_v5.jpg")
        prompt = "person"
        run_v5_pipeline(img_path, json_path, prompt, output_path)

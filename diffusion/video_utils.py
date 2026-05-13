import cv2
import numpy as np
from PIL import Image
import os

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_paths = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()
    return frame_paths

def create_video_from_frames(frame_dir, output_video_path, fps=30):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
    if not frames:
        return
    
    first_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
    height, width, layers = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_name in frames:
        video.write(cv2.imread(os.path.join(frame_dir, frame_name)))
    
    video.release()

def json_to_mask(image_shape, face_data):
    """
    yolo_result.json의 segmentation 데이터를 기반으로 마스크 생성
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for face in face_data.get('faces', []):
        points = np.array(face['segmentation'], dtype=np.int32)
        if len(points) > 0:
            cv2.fillPoly(mask, [points], 255)
    
    kernel_size = 35  # 이 숫자를 키울수록 마스크(인형탈 크기)가 더 커집니다.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return Image.fromarray(mask)

def smooth_masks(masks, kernel_size=5):
    """
    시간적 일관성을 위해 마스크를 부드럽게 처리 (간단한 예시)
    """
    # 실제로는 이전 프레임의 마스크와 현재 마스크를 결합하거나 광학 흐름(Optical Flow)을 사용할 수 있음
    return masks

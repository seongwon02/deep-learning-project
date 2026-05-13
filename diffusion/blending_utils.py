import cv2
import numpy as np
from PIL import Image

def color_transfer(source, target):
    """
    Reinhard et al. 알고리즘을 이용한 색상 전이
    source: 색상을 가져올 원본 이미지 (목/몸 부분 포함)
    target: 색상을 적용할 대상 이미지 (생성된 캐릭터 얼굴)
    """
    source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2LAB).astype("float32")
    target = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2LAB).astype("float32")

    mean_src, std_src = cv2.meanStdDev(source)
    mean_tar, std_tar = cv2.meanStdDev(target)

    # 1D 배열로 평탄화 및 0으로 나누기 방지
    mean_src = mean_src.flatten()
    std_src = std_src.flatten()
    mean_tar = mean_tar.flatten()
    std_tar = std_tar.flatten()
    
    # std가 0인 경우 대비 (매우 작은 값 추가)
    std_tar = np.maximum(std_tar, 1e-5)

    l, a, b = cv2.split(target)
    l = (l - mean_tar[0]) * (std_src[0] / std_tar[0]) + mean_src[0]
    a = (a - mean_tar[1]) * (std_src[1] / std_tar[1]) + mean_src[1]
    b = (b - mean_tar[2]) * (std_src[2] / std_tar[2]) + mean_src[2]

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
    return Image.fromarray(transfer)

def seamless_blend(background, foreground, mask, center):
    """
    OpenCV Poisson Blending을 이용한 자연스러운 합성
    """
    bg_np = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
    fg_np = cv2.cvtColor(np.array(foreground), cv2.COLOR_RGB2BGR)
    mask_np = np.array(mask)
    
    # mask가 3채널이어야 함
    if len(mask_np.shape) == 2:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    
    # Poisson Blending (NORMAL_CLONE 또는 MIXED_CLONE)
    blended = cv2.seamlessClone(fg_np, bg_np, mask_np, center, cv2.NORMAL_CLONE)
    
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
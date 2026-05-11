from ultralytics import YOLO, SAM
import cv2
import json
import os
import numpy as np
import glob

# 1. 모델 로드 (YOLO와 SAM 둘 다 로드)
yolo_model = YOLO('yolov8n-face.pt')
sam_model = SAM('mobile_sam.pt')

# 다운로드한 train 데이터 폴더 경로 설정
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/crowdhuman-face/CrowdHuman Face Voc/train')
image_paths = glob.glob(os.path.join(dataset_dir, '*.jpg'))

if not image_paths:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {dataset_dir}")

# 모든 사진을 돌리면 시간이 너무 오래 걸리므로 (수천 장), 테스트 목적으로 5장만 먼저 처리합니다.
# 전체를 원하시면 image_paths = image_paths[:5] 코드를 지우시면 됩니다.
max_images = 5
image_paths = image_paths[:max_images]

# 2. 결과 저장용 디렉토리 생성
result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result_image')
os.makedirs(result_dir, exist_ok=True)

# face-anonymizer가 요구하는 포맷으로 JSON 뼈대 생성
output_data = {
    "frames": []
}

print(f"총 {len(image_paths)}장의 사진 처리를 시작합니다...")

# 3. 각 사진들에 대해 탐지 및 윤곽선 추출 반복
for frame_index, image_path in enumerate(image_paths):
    print(f"\n[{frame_index+1}/{len(image_paths)}] 처리 중: {os.path.basename(image_path)}")
    
    # 이미지 로드 (한글 경로 호환)
    with open(image_path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        continue
        
    results = yolo_model(img)
    boxes = results[0].boxes
    
    # 현재 프레임에 대한 데이터 딕셔너리
    frame_data = {
        "frame_index": frame_index,
        "image_file": image_path,
        "faces": []
    }
    
    result_img = img.copy()  # 결과를 저장할 이미지 복사
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = round(float(box.conf[0]), 2)
        bbox = [x1, y1, x2, y2]
        
        # SAM 모델을 사용해 윤곽선 추출
        sam_results = sam_model(img, bboxes=bbox, verbose=False)
        
        segmentation_points = []
        if sam_results[0].masks is not None:
            for segment in sam_results[0].masks.xy:
                points = [[int(pt[0]), int(pt[1])] for pt in segment]
                segmentation_points.extend(points)
                
                # 결과 이미지에 윤곽선 그리기
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(result_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 얼굴 정보 (face-anonymizer와 호환되게 'track_id' 사용)
        face_info = {
            "track_id": i + 1,
            "bbox": bbox,
            "confidence": conf,
            "segmentation": segmentation_points
        }
        frame_data["faces"].append(face_info)
        
    output_data["frames"].append(frame_data)

    # 개별 결과 이미지 저장
    result_image_path = os.path.join(result_dir, f'sam_result_{frame_index}.jpg')
    is_success, im_buf_arr = cv2.imencode('.jpg', result_img)
    if is_success:
        im_buf_arr.tofile(result_image_path)

# 4. 종합된 데이터를 JSON 파일로 저장
with open('yolo_result.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

print("\n처리가 완료되었습니다!")
print(f"결과 JSON: yolo_result.json")
print(f"결과 이미지 폴더: {result_dir}")
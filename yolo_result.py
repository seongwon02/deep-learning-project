from ultralytics import YOLO, SAM
import cv2
import json  # 💡 JSON 파일을 다루기 위한 기본 라이브러리 추가

import os
import numpy as np

# 1. 모델 로드 (YOLO와 SAM 둘 다 로드)
yolo_model = YOLO('yolov8n-face.pt')
sam_model = SAM('mobile_sam.pt')

# 이미지 로드 및 YOLO 추론 (한글 경로 문제 해결을 위해 imdecode 사용)
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/test.jpg')
with open(image_path, 'rb') as f:
    img_array = np.frombuffer(f.read(), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없거나 불러올 수 없습니다: {image_path}")
results = yolo_model(img)
boxes = results[0].boxes

# 2. 💡 팀원에게 넘겨줄 데이터를 담을 빈 딕셔너리 생성
output_data = {
    "image_file": image_path,
    "total_faces": len(boxes),
    "faces": []
}

# 3. 탐지된 얼굴의 바운딩 박스를 SAM에 넘겨서 정밀한 윤곽선(마스크) 추출하기
print(f"총 {len(boxes)}개의 얼굴을 발견했습니다. SAM 윤곽선 추출을 시작합니다...")
result_img = img.copy()  # 결과를 저장할 이미지 복사
for i, box in enumerate(boxes):
    # 좌표(소수점 제거)와 신뢰도(소수점 2자리) 추출
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = round(float(box.conf[0]), 2)
    bbox = [x1, y1, x2, y2]
    
    # SAM 모델을 사용해 이 박스 안의 객체(얼굴) 윤곽선 추출
    sam_results = sam_model(img, bboxes=bbox, verbose=False)
    
    segmentation_points = []
    if sam_results[0].masks is not None:
        # masks.xy에 추출된 윤곽선 폴리곤 좌표들이 들어있습니다.
        for segment in sam_results[0].masks.xy:
            # 각 좌표 [x, y]를 정수형 리스트로 변환하여 추가
            points = [[int(pt[0]), int(pt[1])] for pt in segment]
            segmentation_points.extend(points)
            
            # 💡 결과 이미지에 윤곽선 그리기
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 얼굴 정보를 딕셔너리로 묶어서 리스트에 추가 (SAM에서 나온 segmentation 좌표 추가)
    face_info = {
        "id": i + 1,
        "bbox": bbox,
        "confidence": conf,
        "segmentation": segmentation_points  # 💡 정밀한 다각형(Polygon) 좌표
    }
    output_data["faces"].append(face_info)

# 4. 💡 딕셔너리 데이터를 JSON 파일로 저장하기
with open('yolo_result.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

# 5. 💡 결과 이미지를 result_image 디렉토리에 저장하기
result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result_image')
os.makedirs(result_dir, exist_ok=True)
result_image_path = os.path.join(result_dir, 'sam_result.jpg')

# 한글 경로 문제가 있을 수 있으므로 imencode 사용
is_success, im_buf_arr = cv2.imencode('.jpg', result_img)
if is_success:
    im_buf_arr.tofile(result_image_path)
    print(f"SAM 결과 이미지가 저장되었습니다: {result_image_path}")

print("팀원에게 전달할 yolo_result.json 파일이 생성되었습니다! (SAM 좌표 포함)")
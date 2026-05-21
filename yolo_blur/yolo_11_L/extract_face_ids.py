import cv2
import os
from ultralytics import YOLO

def extract_face_ids(video_path, model_path, output_dir, text_file_path):
    print(f"모델 로딩 중: {model_path}")
    model = YOLO(model_path)
    
    # 크롭 이미지를 저장할 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return

    encountered_ids = set()
    
    print("비디오 트래킹 시작... (고유 얼굴 ID 추출 중)")
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 트래킹 수행 (persist=True 로 프레임 간 ID 유지)
        # 높은 정확도를 위해 conf=0.3 이상만 사용
        results = model.track(frame, persist=True, conf=0.3, verbose=False)
        
        # 탐지 결과가 있고, ID가 할당된 경우
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                if track_id not in encountered_ids:
                    encountered_ids.add(track_id)
                    # 바운딩 박스를 정수로 변환
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # 이미지 경계를 벗어나지 않도록 예외 처리
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # 얼굴 영역 크롭
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # 정상적으로 크롭된 경우 저장
                    if face_crop.size != 0:
                        crop_path = os.path.join(output_dir, f"face_ID_{track_id}.jpg")
                        cv2.imwrite(crop_path, face_crop)
                        print(f"-> 새로운 얼굴 발견! ID: {track_id}, 샘플 저장됨")
                        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"진행 상황: {frame_count} / {total_frames} 프레임 분석 완료")

    cap.release()
    
    # 텍스트 파일 생성
    print(f"\n트래킹 완료. 총 {len(encountered_ids)}개의 고유 얼굴을 찾았습니다.")
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write("# 남기고 싶은(비식별화하지 않을) 얼굴의 Action을 'KEEP'으로 변경하세요.\n")
        f.write("# 그 외 'BLUR'로 설정된 ID들은 이후 과정에서 비식별화 처리됩니다.\n")
        f.write(f"# 각 ID의 얼굴 확인용 사진은 '{output_dir}' 폴더를 참조하세요.\n\n")
        
        for track_id in sorted(list(encountered_ids)):
            f.write(f"ID: {track_id}, Action: BLUR\n")
            
    print(f"선택용 텍스트 파일이 정상적으로 생성되었습니다: {text_file_path}")



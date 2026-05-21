import cv2
import os
import json
from ultralytics import YOLO

def read_face_selection(txt_path):
    """
    face_selection.txt 파일을 읽어서 모자이크(BLUR)할 ID와 
    유지할(KEEP) ID를 파싱하여 세트로 반환합니다.
    """
    blur_ids = set()
    keep_ids = set()
    
    if not os.path.exists(txt_path):
        print(f"오류: {txt_path} 파일이 존재하지 않습니다.")
        return blur_ids, keep_ids

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 주석 무시
            if line.startswith('#') or not line:
                continue
                
            # ID: 1, Action: BLUR 파싱
            if 'ID:' in line and 'Action:' in line:
                try:
                    # 간단한 문자열 처리로 추출
                    parts = line.split(',')
                    id_str = parts[0].replace('ID:', '').strip()
                    action_str = parts[1].replace('Action:', '').strip().upper()
                    
                    track_id = int(id_str)
                    if action_str == 'BLUR':
                        blur_ids.add(track_id)
                    elif action_str == 'KEEP':
                        keep_ids.add(track_id)
                except Exception as e:
                    print(f"텍스트 파싱 오류 발생: {line} -> {e}")
                    
    print(f"파싱 완료: {len(blur_ids)}개의 BLUR 대상 ID, {len(keep_ids)}개의 KEEP 대상 ID 확인.")
    return blur_ids, keep_ids

def apply_mosaic(frame, x1, y1, x2, y2, ratio=0.05):
    """지정된 바운딩 박스 영역에 모자이크(픽셀화) 효과를 적용합니다."""
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return frame
        
    h, w = face.shape[:2]
    # 이미지를 작게 축소했다가 원래 크기로 다시 확대하여 픽셀화 효과(모자이크) 생성
    small = cv2.resize(face, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic
    return frame

def apply_anonymization(video_path, model_path, output_video_path, txt_path, json_output_path):
    blur_ids, keep_ids = read_face_selection(txt_path)
    if not blur_ids and not keep_ids:
        print("선택된 ID 정보가 없습니다. 스크립트를 종료합니다.")
        return

    print(f"모델 로딩 중: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # 향후 Diffusion 모델 등에 전달할 목적으로, 프레임별 모자이크 대상 BBox 좌표를 저장하는 딕셔너리
    diffusion_metadata = {}

    print("\n비디오 비식별화 처리 시작...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, persist=True, conf=0.3, verbose=False)
        frame_bboxes = []
        
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # 이미지 경계 예외 처리
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # 사용자가 텍스트 파일에서 BLUR로 지정한 ID만 처리
                if track_id in blur_ids:
                    # 1. 시각적 확인을 위해 모자이크 대신 초록색 Bounding Box 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face ID: {track_id}', (x1, max(y1-10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 2. 나중에 Diffusion 모델로 넘기기 위해 메타데이터 기록
                    frame_bboxes.append({
                        "id": track_id,
                        "bbox": [x1, y1, x2, y2]
                    })
                
                # KEEP으로 지정된 ID는 아무 작업도 하지 않음 (원본 그대로 유지)
        
        if frame_bboxes:
            diffusion_metadata[f"frame_{frame_count}"] = frame_bboxes
            
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"진행 상황: {frame_count} / {total_frames} 프레임 처리 완료")

    cap.release()
    out.release()
    
    # Diffusion용 텍스트/JSON 파일 저장
    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump(diffusion_metadata, jf, indent=4)
        
    print(f"\n처리가 완료되었습니다!")
    print(f"비식별화 영상 저장: {output_video_path}")
    print(f"Diffusion 모델 전달용 BBox 정보 저장: {json_output_path}")

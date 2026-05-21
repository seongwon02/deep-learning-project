import cv2
import os
import json
from ultralytics import YOLO

# 기존 모듈 함수 임포트
from extract_face_ids import extract_face_ids
from apply_anonymization import read_face_selection

def get_all_detected_ids(txt_path):
    """face_selection.txt 파일에 나열된 모든 ID 목록을 파싱하여 반환합니다."""
    all_ids = set()
    if not os.path.exists(txt_path):
        return all_ids
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if 'ID:' in line:
                try:
                    parts = line.split(',')
                    id_str = parts[0].replace('ID:', '').strip()
                    all_ids.add(int(id_str))
                except Exception:
                    pass
    return all_ids

def write_face_selection(txt_path, blur_ids, keep_ids):
    """업데이트된 blur_ids와 keep_ids를 face_selection.txt에 다시 기록하여 동기화합니다."""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("# 남기고 싶은(비식별화하지 않을) 얼굴의 Action을 'KEEP'으로 변경하세요.\n")
        f.write("# 그 외 'BLUR'로 설정된 ID들은 이후 과정에서 비식별화 처리됩니다.\n")
        f.write("# 각 ID의 얼굴 확인용 사진은 'extracted_faces' 폴더를 참조하세요.\n\n")
        
        all_ids = sorted(list(blur_ids | keep_ids))
        for track_id in all_ids:
            action = "KEEP" if track_id in keep_ids else "BLUR"
            f.write(f"ID: {track_id}, Action: {action}\n")

def run_pipeline(video_path, model_path, output_dir, text_file_path, json_output_path, output_video_path=None):
    # ----------------------------------------------------
    # 1단계: 객체 인식 및 트래킹 -> 고유 ID별 얼굴 이미지 크롭 -> face_selection.txt 작성
    # ----------------------------------------------------
    print("\n=== [1단계] 얼굴 ID 추출 및 샘플 크롭 이미지 생성 ===")
    extract_face_ids(video_path, model_path, output_dir, text_file_path)
    
    # ----------------------------------------------------
    # 사용자 작업 대기: 콘솔 입력 또는 face_selection.txt 수정을 대기
    # ----------------------------------------------------
    print(f"\n[대기] '{text_file_path}' 파일이 생성되었습니다.")
    print("       'extracted_faces' 폴더의 이미지를 확인한 후:")
    print("       1. KEEP(비식별화 제외) 처리할 얼굴 ID들을 공백으로 구분하여 직접 입력창에 입력해 주세요. (예: 5 8 90 420)")
    print("       2. 또는 직접 'face_selection.txt' 파일을 메모장 등으로 열어 편집하셔도 됩니다.")
    print("       >> 아무것도 입력하지 않고 Enter만 누르면 기존 'face_selection.txt' 파일 설정대로 처리됩니다.")
    
    user_input = input("KEEP ID 입력 >> ").strip()
    
    # ----------------------------------------------------
    # 2단계: 입력 결과에 따라 설정 파싱 및 동기화
    # ----------------------------------------------------
    print("\n=== [2단계] 사용자 설정 파싱 및 동기화 ===")
    
    if user_input:
        # 콘솔 입력을 통해 KEEP 처리할 ID들을 전달받은 경우
        try:
            keep_ids = set(int(x) for x in user_input.split())
            all_ids = get_all_detected_ids(text_file_path)
            
            # 입력한 ID 중 감지되지 않은 ID가 있으면 경고 메시지 출력
            invalid_ids = keep_ids - all_ids
            if invalid_ids:
                print(f"[경고] 감지되지 않은 잘못된 ID가 입력에 포함되었습니다: {invalid_ids}")
            
            # 유효한 ID들만 필터링
            keep_ids = keep_ids & all_ids
            blur_ids = all_ids - keep_ids
            
            # 텍스트 파일에 변경 내용 동기화 기록
            write_face_selection(text_file_path, blur_ids, keep_ids)
            print(f"콘솔 입력 반영 완료: {len(keep_ids)}개 KEEP 지정 (IDs: {sorted(list(keep_ids))})")
            print(f"나머지 {len(blur_ids)}개 ID는 BLUR(BBox 대상)로 자동 지정되었습니다.")
        except ValueError:
            print("[오류] 입력 형식이 올바르지 않습니다. 공백으로 구분된 숫자여야 합니다. (예: 5 8 90)")
            print("기존 'face_selection.txt' 파일의 설정을 불러옵니다...")
            blur_ids, keep_ids = read_face_selection(text_file_path)
    else:
        # 콘솔 입력이 없고 그냥 Enter만 누른 경우 -> 기존 txt 파일 그대로 파싱
        print("콘솔 입력이 없어 기존 'face_selection.txt' 설정을 사용합니다.")
        blur_ids, keep_ids = read_face_selection(text_file_path)
        
    if not blur_ids and not keep_ids:
        print("[오류] 선택된 ID 정보가 없습니다. 파이프라인을 종료합니다.")
        return

    # ----------------------------------------------------
    # 3단계: 비디오 트래킹 및 Diffusion 전달용 BBox JSON 데이터 추출
    # ----------------------------------------------------
    print("\n=== [3단계] 비디오 분석 및 BBox JSON 데이터 생성 ===")
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
    
    print(f"비디오 정보: 해상도 {width}x{height}, FPS: {fps}, 총 프레임 수: {total_frames}")

    # [결과 동영상 렌더링 설정 - 필요 시 아래 주석 해제]
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    diffusion_metadata = {}
    
    # 5프레임 간격 탐지를 위한 설정 및 임시 저장소
    keyframe_interval = 5
    current_tracks = [] # [{'id': track_id, 'bbox': [x1, y1, x2, y2]}, ...]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 5프레임마다 한 번씩만 YOLO 트래킹 수행
        if frame_count % keyframe_interval == 0:
            results = model.track(frame, persist=True, conf=0.3, verbose=False)
            current_tracks = []
            
            if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # 이미지 경계 예외 처리
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    current_tracks.append({
                        "id": track_id,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        # 매 프레임별로 현재 저장된 트랙 정보(직전 keyframe의 탐지결과)를 사용해 BBox 기록
        frame_bboxes = []
        for track in current_tracks:
            track_id = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            
            # 텍스트 설정에서 BLUR로 지정한 ID만 수집
            if track_id in blur_ids:
                frame_bboxes.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2]
                })
                
                # [결과 동영상 렌더링 - 필요 시 아래 주석 해제]
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, f'Face ID: {track_id}', (x1, max(y1-10, 10)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if frame_bboxes:
            diffusion_metadata[f"frame_{frame_count}"] = frame_bboxes
            
        # [결과 동영상 렌더링 - 필요 시 아래 주석 해제]
        # out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"진행 상황: {frame_count} / {total_frames} 프레임 분석 완료")

    cap.release()
    # [결과 동영상 렌더링 해제 - 필요 시 아래 주석 해제]
    # out.release()
    
    # Diffusion용 JSON 파일 저장
    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump(diffusion_metadata, jf, indent=4)
        
    print(f"\n[완료] Diffusion 모델 전달용 BBox 정보 저장: {json_output_path}")
    
    # [결과 동영상 렌더링 완료 메시지 - 필요 시 아래 주석 해제]
    # print(f"[완료] 결과 가공 영상 저장: {output_video_path}")

if __name__ == '__main__':
    # 기본 경로 설정
    video_file = '1643-148614430.mp4'
    model_file = 'best.pt'
    
    current_dir = os.path.basename(os.getcwd())
    if current_dir != 'yolo_11_L':
        # 상위 폴더(yolo_blur)에서 실행할 경우 경로 자동 보정
        video_file = os.path.join('yolo_11_L', video_file)
        model_file = os.path.join('yolo_11_L', model_file)
        output_dir = os.path.join('yolo_11_L', 'extracted_faces')
        text_file = os.path.join('yolo_11_L', 'face_selection.txt')
        json_output = os.path.join('yolo_11_L', 'blur_bboxes_for_diffusion.json')
        output_video = os.path.join('yolo_11_L', 'result_1643_anonymized.mp4')
    else:
        output_dir = 'extracted_faces'
        text_file = 'face_selection.txt'
        json_output = 'blur_bboxes_for_diffusion.json'
        output_video = 'result_1643_anonymized.mp4'
        
    run_pipeline(
        video_path=video_file,
        model_path=model_file,
        output_dir=output_dir,
        text_file_path=text_file,
        json_output_path=json_output,
        output_video_path=output_video
    )

# YOLOv11 얼굴 탐지 및 비식별화 BBox 추출 모듈 (yolo_11_L)

본 모듈은 비디오에서 인물별 얼굴을 정밀 탐지하여 고유 ID를 부여하고, 사용자가 지정한 비식별화(BBox 타겟팅) 대상의 프레임별 좌표 목록을 생성형 AI(Diffusion) 처리 용도로 추출하는 역할을 수행합니다.

---

## 📂 폴더 파일 구성

* **[main.py](main.py)**: 전체 전처리 파이프라인의 진입점(Entry Point) 실행 파일
* **[extract_face_ids.py](extract_face_ids.py)**: 고유 ID 판별 및 크롭 이미지 추출 모듈
* **[apply_anonymization.py](apply_anonymization.py)**: 텍스트 설정 파일 파싱 및 비식별화 도우미 함수 모듈
* **best.pt**: 학습 완료된 YOLOv11 Large 얼굴 탐지 모델 가중치 파일
* **extracted_faces/**: 동영상 내에서 감지되어 고유 ID별로 크롭된 얼굴 샘플 이미지 저장 폴더 (자동 생성)
* **face_selection.txt**: 비식별화 및 유지할 얼굴 ID 설정 파일 (자동 생성)
* **blur_bboxes_for_diffusion.json**: 최종 결과물인 프레임별 타겟 BBox 좌표 JSON 파일 (자동 생성)

---

## 🚀 파이프라인 작동 흐름 및 실행 방법

`yolo_blur/yolo_blur` 경로를 작업 디렉토리로 하여 터미널에서 아래 명령을 실행합니다.

```bash
python yolo_11_L/main.py
```

### 1단계: 고유 ID별 얼굴 샘플 추출
* 비디오 전체를 분석하여 처음 등장하는 인물의 얼굴 영역을 `extracted_faces/face_ID_<id>.jpg` 파일로 각각 저장합니다.
* 전체 고유 ID 목록을 포함한 `face_selection.txt` 템플릿 파일을 생성합니다.

### 2단계: 사용자 인터랙티브 설정 (대기)
콘솔 창에서 아래와 같은 입력 프롬프트가 실행되며 진행이 멈춥니다.
```text
KEEP ID 입력 >> 
```
* **콘솔 입력 방식 (추천)**:
  `extracted_faces` 폴더 내 크롭 이미지들을 확인한 뒤, **비식별화 대상에서 제외할(원본을 유지할) 얼굴 ID들**을 공백으로 구분해 직접 타이핑합니다. (예: `5 8 90 420`)
  입력 즉시 해당 ID들은 `KEEP`으로, 입력하지 않은 나머지 전체 ID들은 `BLUR`로 자동 매핑되어 `face_selection.txt` 파일에 즉시 동기화되어 기록됩니다.
* **텍스트 파일 수정 방식**:
  콘솔에 아무것도 입력하지 않고 `face_selection.txt` 파일을 직접 열어 대상 ID의 Action 필드를 `KEEP` 또는 `BLUR`로 직접 변경한 뒤, 콘솔창에서 `[Enter]` 키만 눌러도 설정값이 읽혀 다음 단계가 진행됩니다.

### 3단계: Keyframe 최적화 트래킹 및 JSON 데이터 추출
* **Keyframe 적용 (5프레임 주기)**: 실시간 연산량 최적화를 위해 YOLO 트래킹(`model.track`)은 5프레임 간격마다 구동됩니다. 나머지 구간 프레임에서는 직전 keyframe의 탐지결과(`current_tracks`) 데이터를 캐싱하여 사용하므로, 트래킹 ID의 일관성은 유지되면서 동작 연산 부하를 약 1/5로 경감시켰습니다.
* **최종 JSON 파일 생성**: `BLUR`로 분류된 비식별화 타겟들의 프레임별 바운딩 박스(BBox) 좌표 정보가 `blur_bboxes_for_diffusion.json` 파일에 저장됩니다.

---

## 🎨 결과 동영상 시각화 렌더링 방법
기본적으로 Diffusion 전송을 위한 JSON 메타데이터 추출이 주 목적이므로 동영상 가공 렌더링 코드는 주석 처리되어 있습니다.
만약 얼굴 BBox 영역 및 ID 라벨이 비디오 화면에 표시된 최종 결과물 영상(`.mp4`)을 시각적으로 확인하고 싶으시다면, [main.py](main.py) 파일 내부에서 아래의 주석들을 해제하고 다시 실행하십시오.

* `out = cv2.VideoWriter(...)` 관련 정의 부분 주석 해제
* `cv2.rectangle` 및 `cv2.putText` 오버레이 그리기 부분 주석 해제
* `out.write(frame)` 프레임 출력 부분 주석 해제
* `out.release()` 비디오 저장 완료 부분 주석 해제

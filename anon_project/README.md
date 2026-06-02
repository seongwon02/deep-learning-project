# 🎭 Face Anonymization Modular Pipeline (`anon_project`)

이 프로젝트는 동영상 및 사진 속 인물들의 얼굴을 탐지하고, 선택적으로 본인(크리에이터)을 보호하면서 제3자의 얼굴만 안전하게 비식별화하는 **모듈식 하이브리드 파이프라인**입니다. 
기존의 스크립트 기반 코드를 모듈별 폴더로 분리하고, 통합 Streamlit 웹앱(`app.py`)이 이 모듈들을 직접 `import`하여 구동하도록 깔끔하게 리팩토링되었습니다.

---

## 📁 디렉토리 구조 및 역할

```
anon_project/
├── __init__.py
├── app.py                      # 모듈식 통합 Streamlit 웹 애플리케이션
├── yolo/
│   ├── detector.py             # YOLOv8/11 기반 얼굴 검출 & 트래킹 API
│   └── crop.py                 # 얼굴 Crop 및 InsightFace 검증 API
├── blur/
│   └── processor.py            # Gaussian Blur 및 Pixelation(모자이크) API
├── sticker/
│   ├── processor.py            # 스티커 & 이모지 합성 API
│   └── sticker_anonymizer/     # 스티커 렌더링 엔진 코어
├── object3d/
│   └── processor.py            # 3D 오브젝트(헬멧 등) 렌더링 & 스무딩 API
└── diffusion/
    ├── processor.py            # 디퓨전(InstantID/Face Blend) AI 가상화 API
    └── ...                     # SDXL InstantID 렌더러 등 관련 핵심 구현체
```

---

## ⚙️ 필요한 가중치 파일 (Weights/Models)

모듈이 정상 구동하기 위해 아래 가중치 파일들이 올바른 경로에 배치되어 있어야 합니다.

### 1. YOLOv8 Face Tracking 모델
* **파일명**: `face_yolov8n.pt` 또는 `yolo11l.pt`
* **추천 경로**: `real/` (또는 프로젝트 최상위 루트)
* **역할**: 비디오/이미지에서 얼굴 바운딩 박스(bbox) 및 5대 랜드마크(눈, 코, 입꼬리)를 빠르고 정확하게 검출하고 트래킹(추적)합니다.

### 2. InsightFace Buffalo_l 모델
* **파일명/폴더**: `models/buffalo_l/` 하위 ONNX 파일들
  * (예: `det_10g.onnx`, `2d106det.onnx`, `1k3d68.onnx`, `w600k_r50.onnx`, `genderage.onnx`)
* **추천 경로**: `real/models/buffalo_l/` (또는 `models/buffalo_l/`)
* **역할**: 
  1. **YOLO Crop 검증**: YOLO가 오검출한 가짜 얼굴(노이즈)을 필터링하기 위해 크로스-벨리데이션을 진행합니다.
  2. **3D Helmet Pose Estimation**: 얼굴의 정확한 3차원 회전 각도(pitch, yaw, roll)를 추출하여 3D 헬멧을 각도에 맞게 렌더링합니다.

### 3. SAM 2.1 (Segment Anything) 모델
* **파일명**: `sam2.1_l.pt`
* **추천 경로**: 프로젝트 루트 또는 `real/` 디렉토리
* **역할**: Generative Diffusion 모드에서 정밀한 얼굴 경계 마스크를 생성할 때 활용됩니다.

### 4. 미디어 및 3D 에셋 (OBJ, PNG 등)
* **주의**: `anon_project/` 리포지토리 내부에는 깃허브 업로드 최적화 및 순수 코드 보존을 위해 3D OBJ 모델, PNG 스티커 이미지 등의 바이너리 자산이 포함되어 있지 않습니다.
* **사용 시**: 스크립트 실행 또는 API 호출 시 에셋이 위치한 외부 경로를 명시해야 합니다. (예: `sticker_png_path="real/sticker/assets/panda.png"`, `obj_path="real/helmet/10517_Motorcycle_Helmet_v01_L3.obj"`)

---

## 🚀 사용 방법

### 1. Streamlit 웹 애플리케이션 실행
터미널에서 아래 명령을 실행하여 고품질의 비식별화 편집 스튜디오를 브라우저에 띄울 수 있습니다.
```bash
# deep-learning-project 폴더 위치에서 실행
streamlit run real/anon_project/app.py
```
* **기능**: 사진/동영상 업로드 ➡️ YOLO 검출 ➡️ 보호할 인물(예: 크리에이터 ID) 선택 ➡️ 블러, 스티커, 3D 헬멧, 디퓨전 방식 적용 및 다운로드.

---

### 2. Python 코드로 직접 가져다 쓰기 (API 활용)

다른 파이썬 스크립트나 웹 프레임워크(FastAPI 등) 개발 시 다음과 같이 각 모듈의 API를 직접 임포트하여 파이프라인을 자유롭게 구성할 수 있습니다.

#### ① YOLO 트래킹 & 검출
```python
from anon_project.yolo import run_tracking

# 비디오에서 얼굴을 트래킹하고 감지 결과를 딕셔너리로 추출 (JSON 저장 가능)
detections = run_tracking(
    input_path="input_video.mp4", 
    output_json_path="detections.json",
    conf_threshold=0.3
)
```

#### ② 3D Helmet Overlay (3D 헬멧 비식별화)
```python
from anon_project.object3d import apply_3d_anonymization

apply_3d_anonymization(
    input_path="input_video.mp4",
    output_path="output_helmet.mp4",
    detections="detections.json",       # JSON 파일 경로 혹은 Dict 전달 가능
    keep_track_ids={1},                # 1번 인물은 헬멧을 씌우지 않고 보호
    helmet_scale=1.35,
    y_shift=-0.15,
    z_shift=-6.0
)
```

#### ③ Blur & Pixelation (블러 / 모자이크 비식별화)
```python
from anon_project.blur import apply_blur_anonymization

apply_blur_anonymization(
    input_path="input_video.mp4",
    output_path="output_blur.mp4",
    detections="detections.json",
    keep_track_ids={1},
    fallback_mode="pixelate",          # "blur"(블러) 또는 "pixelate"(모자이크)
    fallback_pixel_size=8              # 모자이크 픽셀 크기
)
```

#### ④ Sticker & Emoji (스티커 이미지 / 이모지 비식별화)
```python
from anon_project.sticker import apply_sticker_anonymization

apply_sticker_anonymization(
    input_path="input_video.mp4",
    output_path="output_sticker.mp4",
    detections="detections.json",
    keep_track_ids={1},
    emoji_char="🐼",                    # PNG 스티커가 없을 때 대체 적용할 이모지
    sticker_png_path="panda.png"        # 스티커 이미지 이름 또는 절대경로 (생략 시 이모지 적용)
)
```

#### ⑤ Generative Diffusion (AI 생성형 비식별화)
```python
from anon_project.diffusion import apply_diffusion_anonymization

apply_diffusion_anonymization(
    input_path="input_image.png",
    output_path="output_diffusion.png",
    detections="detections.json",
    keep_track_ids={1},
    ref_path="reference_face.png",      # 대체할 타겟 스타일 얼굴 이미지
    ref_mode="Face Blend"              # "Face Blend", "InstantID/IP-Adapter", "Prompt Only"
)
```

## 사용방법
### 1. 의존성 패키지 설치
============================================
%cd /content/deep-learning-project

!pip install -r yolo_blur/requirements.txt
!pip install -r diffusion/requirements.txt
!pip install -U torchao
!pip install -U xformers
============================================

### 2. yolov8n-face.pt 모델 다운로드
============================================
%cd /content/deep-learning-project

!wget https://huggingface.co/WePrompt/deepface-weights/resolve/main/yolov8n-face.pt
============================================

### 3. 
============================================
!python yolo_blur/yolo_result.py
============================================

### 4
============================================
!python diffusion/main_pipeline.py
============================================

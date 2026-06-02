"""
Modular Face Anonymizer Streamlit UI.
Imports and runs Python APIs directly from submodules inside anon_project:
  - yolo: Face detection & tracking
  - blur: Gaussian Blur / Pixelation
  - sticker: Emoji / PNG Sticker overlay
  - 3dobject: 3D OBJ (Helmet) overlay
  - diffusion: Generative Diffusion XL inpainting
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Ensure the parent directory of anon_project is in sys.path to resolve imports correctly
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import modular APIs directly
from anon_project.yolo import run_tracking, crop_and_validate_faces
from anon_project.blur import apply_blur_anonymization
from anon_project.sticker import apply_sticker_anonymization
from anon_project.object3d import apply_3d_anonymization
from anon_project.diffusion import apply_diffusion_anonymization

st.set_page_config(page_title="Face Anonymizer Studio", page_icon="🎭", layout="wide")

# ──────────────────────────────────────────────
# PREMIUM STYLING (CSS Injections)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font Override */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Accent Gradient */
    .title-gradient {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.25rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Glassmorphic Card Style */
    .pitch-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2.0rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .pitch-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 1.0rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Comparison Table Style */
    .comp-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-size: 1.0rem;
    }
    
    .comp-table th {
        background: rgba(168, 85, 247, 0.15);
        color: #e9d5ff;
        font-weight: 600;
        text-align: left;
        padding: 12px;
        border-bottom: 2px solid rgba(168, 85, 247, 0.3);
    }
    
    .comp-table td {
        padding: 14px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        color: #d1d5db;
    }
    
    .comp-table tr:hover {
        background: rgba(255, 255, 255, 0.02);
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    .badge-diffusion {
        background-color: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-blur {
        background-color: rgba(168, 85, 247, 0.2);
        color: #f5d0fe;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2.0rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
        font-size: 1.05rem !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(168, 85, 247, 0.45) !important;
        background: linear-gradient(135deg, #9333ea 0%, #2563eb 100%) !important;
    }
    
    /* File Uploader Custom Aesthetics */
    section[data-testid="stFileUploader"] {
        border: 2px dashed rgba(168, 85, 247, 0.3) !important;
        border-radius: 16px !important;
        background-color: rgba(168, 85, 247, 0.01) !important;
        padding: 1.0rem !important;
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stFileUploader"]:hover {
        border-color: #a855f7 !important;
        background-color: rgba(168, 85, 247, 0.04) !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DETECTIONS PARSING & HELPERS
# ──────────────────────────────────────────────

def parse_detections(data: dict | list) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
        if not frames:
            return []
        first = frames[next(iter(frames))] if isinstance(frames, dict) else frames[0]
        if isinstance(first, dict):
            for key in ("faces", "detections", "objects", "tracks"):
                if key in first:
                    return first[key]
        return first if isinstance(first, list) else []
    for key in ("faces", "detections", "0"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []

def get_bbox(face: dict) -> tuple[int, int, int, int] | None:
    raw = face.get("bbox") or face.get("box") or face.get("xyxy")
    if raw is None:
        if all(k in face for k in ("x1", "y1", "x2", "y2")):
            return int(face["x1"]), int(face["y1"]), int(face["x2"]), int(face["y2"])
        if all(k in face for k in ("x", "y", "w", "h")):
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            return int(x), int(y), int(x + w), int(y + h)
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
    return None

def get_track_id(face: dict) -> str | None:
    for key in ("track_id", "trackId", "id", "track"):
        if key in face and face[key] is not None:
            return str(face[key])
    return None

def extract_all_track_ids(normalized_data: dict) -> list[str]:
    ids = set()
    for frame in normalized_data.get("frames", []):
        for face in frame.get("faces", []):
            tid = get_track_id(face)
            if tid:
                ids.add(tid)
    try:
        return sorted(list(ids), key=lambda x: int(x))
    except ValueError:
        return sorted(list(ids))

def crop_face(image: Image.Image, bbox: tuple, pad: float = 0.3) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(w * pad))
    y1 = max(0, y1 - int(h * pad))
    x2 = min(image.width, x2 + int(w * pad))
    y2 = min(image.height, y2 + int(h * pad))
    return image.crop((x1, y1, x2, y2)).resize((150, 150), Image.Resampling.LANCZOS)

def draw_bboxes(image: Image.Image, detections: list, keep_ids: set) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for face in detections:
        bbox = get_bbox(face)
        tid = get_track_id(face)
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        color = (34, 197, 94) if tid in keep_ids else (239, 68, 68) # green-500 or red-500
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"ID {tid} {'KEEP' if tid in keep_ids else 'ANON'}"
        draw.text((x1, max(0, y1 - 18)), label, fill=color, font=font)
    return out

def get_first_frame(path: str) -> Image.Image | None:
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None

# ──────────────────────────────────────────────
# SESSION STATES
# ──────────────────────────────────────────────
defaults = {
    "input_path": None, "det_path": None,
    "detections": [], "all_track_ids": [], "preview": None,
    "first_frame_faces": [],
    "keep_ids": set(), "output_path": None,
    "work_dir": tempfile.mkdtemp(prefix="face_anon_modular_"),
    "ref_path": None, "ref_preview": None,
    "custom_sticker_path": None, "custom_sticker_preview": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

WORK = Path(st.session_state.work_dir)

# ──────────────────────────────────────────────
# HEADER RENDERING
# ──────────────────────────────────────────────
st.markdown('<div class="title-gradient">🎭 Face Anonymizer Studio (Modular)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">리팩토링된 모듈식 얼굴 비식별화 및 크리에이터 보호를 위한 하이브리드 파이프라인</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ANONYMIZER STUDIO (PLAYGROUND)
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 미디어 업로드")
    
    media_type = st.radio("미디어 분류", ["사진 (Image)", "동영상 (Video)"], horizontal=True)
    
    if media_type == "사진 (Image)":
        accept_types = ["jpg", "jpeg", "png", "webp", "bmp"]
    else:
        accept_types = ["mp4", "mov", "avi", "mkv", "webm"]
        
    media_file = st.file_uploader(
        f"원본 {media_type} 파일 선택",
        type=accept_types,
        key="studio_media_uploader"
    )
    
    if media_file:
        p = WORK / media_file.name
        p.write_bytes(media_file.getbuffer())
        
        if st.session_state.input_path != str(p):
            st.session_state.input_path = str(p)
            st.session_state.det_path = None
            st.session_state.detections = []
            st.session_state.all_track_ids = []
            st.session_state.preview = None
            st.session_state.output_path = None
        
        ext = p.suffix.lower()
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            if st.session_state.preview is None:
                st.session_state.preview = get_first_frame(str(p))
            st.video(str(p))
        else:
            if st.session_state.preview is None:
                st.session_state.preview = Image.open(p).convert("RGB")
            st.image(st.session_state.preview, use_container_width=True)
            
    # 3. YOLO 얼굴 검출 작동 (Direct Call)
    if st.session_state.input_path and not st.session_state.det_path:
        with st.spinner("🤖 YOLO 기반 객체 탐지 및 얼굴 추적(Tracking) 수행 중..."):
            try:
                # Resolve YOLO model path relative to project
                yolo_model = project_root / "yolov8n-face.pt"
                if not yolo_model.exists():
                    yolo_model = project_root / "yolo_blur" / "yolo_11_L" / "best.pt"
                    if not yolo_model.exists():
                        yolo_model = project_root / "yolo11l.pt"
                    
                # Call modular run_tracking directly
                normalized = run_tracking(st.session_state.input_path, model_path=yolo_model)
                
                p = WORK / "auto_detections.json"
                p.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
                st.session_state.det_path = str(p)
                
                with st.spinner("🔍 InsightFace 교차 검증 중 (오탐 필터링)..."):
                    validation_results = crop_and_validate_faces(
                        input_path=st.session_state.input_path,
                        detections=normalized,
                        output_dir=WORK / "extracted_faces",
                        cross_validate=True
                    )
                    
                unique_faces = []
                for tid, meta in validation_results.get("faces", {}).items():
                    unique_faces.append({
                        "track_id": str(tid),
                        "crop_path": meta["crop_path"]
                    })
                    
                st.session_state.detections = unique_faces
                st.session_state.first_frame_faces = parse_detections(normalized)
                st.session_state.all_track_ids = extract_all_track_ids(normalized)
                st.rerun()
            except Exception as e:
                st.error(f"얼굴 검출 중 오류 발생: {e}")
                
    # 4. 검출된 얼굴 리스트 & 보호 ID 설정
    dets = st.session_state.detections
    preview = st.session_state.preview
    
    if dets and st.session_state.det_path:
        st.divider()
        st.markdown("### 🔍 탐지된 인물 분석 및 보호 대상 지정")
        st.markdown("이 영상/사진 속에서 **비식별화하지 않고 그대로 보호할 인물(크리에이터 본인)**을 선택하세요.")
        
        # Crop 갤러리 그리드 표시
        COLS = 5
        rows = [dets[i:i+COLS] for i in range(0, len(dets), COLS)]
        for row in rows:
            cols = st.columns(len(row))
            for col, face in zip(cols, row):
                tid = face.get("track_id")
                crop_path = face.get("crop_path")
                with col:
                    if crop_path and os.path.exists(crop_path):
                        try:
                            img = Image.open(crop_path).convert("RGB")
                            st.image(img, use_container_width=True)
                        except Exception:
                            pass
                        
                        is_protected = tid in st.session_state.keep_ids
                        if is_protected:
                            st.markdown(f"<p style='text-align:center;color:#2ecc71;font-weight:bold;margin:2px;font-size:13px'>🛡️ ID {tid} (보호됨)</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p style='text-align:center;color:#e74c3c;font-weight:bold;margin:2px;font-size:13px'>🎭 ID {tid} (대상)</p>", unsafe_allow_html=True)
                            
        # ID 입력 필드
        all_ids_str = ", ".join(st.session_state.all_track_ids)
        st.caption(f"검출된 모든 인물 Face ID 목록: **{all_ids_str}**")
        
        keep_input = st.text_input(
            "보호할(원본 유지할) ID 번호 입력 (쉼표 구분)",
            value=",".join(sorted(st.session_state.keep_ids)),
            placeholder="예: 1, 3"
        )
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            if st.button("💾 적용하기", key="btn_apply_keep_ids"):
                st.session_state.keep_ids = {x.strip() for x in keep_input.split(",") if x.strip()}
                st.rerun()
        with col_k2:
            if st.button("🎭 전체 비식별화", key="btn_clear_keep_ids"):
                st.session_state.keep_ids = set()
                st.rerun()
                
        # 5. 상태 시각화 박스 오버레이 그리기
        first_frame_faces = st.session_state.get("first_frame_faces", [])
        if preview:
            st.image(draw_bboxes(preview, first_frame_faces, st.session_state.keep_ids),
                     caption="초록색(🛡️)=보호 대상, 빨간색(🎭)=비식별화 진행 대상", use_container_width=True)

with col_right:
    st.markdown("### ⚙️ 비식별화 설정")
    
    # 1. 처리 방식 선택
    if media_type == "사진 (Image)":
        method = st.selectbox("아노니마이징 기법", ["블러 / 픽셀화", "스티커 이미지 붙이기", "Generative Diffusion (AI 생성)"])
    else:
        method = st.selectbox("아노니마이징 기법", ["블러 / 픽셀화", "스티커 이미지 붙이기", "3D Helmet Overlay (3D 헬멧 씌우기)", "Generative Diffusion (AI 생성)"])
        
    # 2. 기법별 세부 옵션들
    if method == "Generative Diffusion (AI 생성)":
        st.warning("⚠️ 디퓨전 생성 기법은 높은 연산량을 필요로 하므로 작동 시 수 분의 시간이 걸릴 수 있습니다. (GPU 환경 권장)")
        
        diff_mode = st.selectbox("레퍼런스 스타일 방식", ["얼굴 합성 (Face Blend)", "아이덴티티 보존 (InstantID/IP-Adapter)", "프롬프트 추출 (Prompt Only)"])
        
        ref_uploader = st.file_uploader(
            "스타일 대체의 기준이 될 레퍼런스 얼굴 사진 (선택)",
            type=["jpg", "jpeg", "png", "webp"],
            key="studio_ref_uploader"
        )
        if ref_uploader:
            ref_p = WORK / ref_uploader.name
            ref_p.write_bytes(ref_uploader.getbuffer())
            st.session_state.ref_path = str(ref_p)
            st.session_state.ref_preview = Image.open(ref_p).convert("RGB")
            st.image(st.session_state.ref_preview, caption="등록된 레퍼런스 스타일", width=200)
            
        inpaint_scope = st.selectbox("생성 경계 범위", ["face-crop", "full-frame"])
        mask_mode = st.selectbox("마스크 따기 방식", ["sam", "ellipse", "bbox", "auto"])
        seed = st.number_input("재현성 시드(Seed)", value=1234, min_value=0)
        max_frames = st.number_input("최대 처리 프레임 (0 = 전체)", value=15, min_value=0)
        
    elif method == "스티커 이미지 붙이기":
        sticker_type = st.radio("스티커 종류", ["기본 탑재 이미지 스티커", "이모지 문자 입력", "커스텀 스티커 이미지 업로드"])
        
        emoji_char = "🐼"
        custom_sticker_path = None
        
        if sticker_type == "기본 탑재 이미지 스티커":
            default_sticker_select = st.selectbox("기본 스티커 선택", ["Panda Mask (판다 가면)", "Helmet Mask (헬멧)"])
            if default_sticker_select == "Panda Mask (판다 가면)":
                custom_sticker_path = "panda.png"
            else:
                custom_sticker_path = "helmet_ref.png"
                
        elif sticker_type == "이모지 문자 입력":
            emoji_char = st.text_input("사용할 이모지 문자력", value="🐼")
            
        else:
            custom_uploader = st.file_uploader(
                "커스텀 투명 배경(PNG) 스티커 업로드",
                type=["png"],
                key="studio_custom_sticker_uploader"
            )
            if custom_uploader:
                st_p = WORK / custom_uploader.name
                st_p.write_bytes(custom_uploader.getbuffer())
                st.session_state.custom_sticker_path = str(st_p)
                st.session_state.custom_sticker_preview = Image.open(st_p).convert("RGBA")
                st.image(st.session_state.custom_sticker_preview, caption="업로드한 커스텀 스티커", width=120)
                custom_sticker_path = st.session_state.custom_sticker_path
                
        with st.expander("⭐ 스티커 크기 및 적용 옵션"):
            sticker_min_face_size = st.slider(
                "최소 얼굴 크기 (픽셀)", 
                min_value=0, max_value=100, value=40, step=5,
                help="이 크기보다 작은 얼굴은 스티커 대신 모자이크 처리됩니다. 항상 스티커를 적용하려면 0으로 낮추세요."
            )
        
        max_frames = st.number_input("최대 처리 프레임 (0 = 전체)", value=0, min_value=0)
        
    elif method == "3D Helmet Overlay (3D 헬멧 씌우기)":
        st.info("🏍️ YOLO tracking 바운딩박스 + InsightFace Pose(회전 각도) 데이터를 추출해 자연스러운 3D 헬멧 오버레이를 진행합니다.")
        
        helmet_scale = st.slider("헬멧 크기 배율 (Helmet Scale)", min_value=0.5, max_value=2.0, value=1.35, step=0.05)
        y_shift = st.slider("세로 오프셋 (Y-Shift, 음수=위로)", min_value=-0.5, max_value=0.5, value=-0.15, step=0.01)
        z_shift = st.slider("가로/깊이 오프셋 (Z-Shift, 음수=뒤로)", min_value=-15.0, max_value=0.0, value=-6.0, step=0.5)
        
        with st.expander("⭐ 플리커링(떨림) 스무딩 & Fallback 옵션"):
            size_deadband = st.slider("크기 변화 데드밴드 (Size Deadband)", min_value=0.0, max_value=0.20, value=0.08, step=0.01)
            pos_deadband = st.slider("위치 변화 데드밴드 (Position Deadband)", min_value=0.0, max_value=0.10, value=0.04, step=0.01)
            ema_size = st.slider("크기 스무딩 (EMA size, 작을수록 부드러움)", min_value=0.05, max_value=1.0, value=0.30, step=0.05)
            ema_pos = st.slider("위치 스무딩 (EMA position, 작을수록 부드러움)", min_value=0.05, max_value=1.0, value=0.30, step=0.05)
            ema_pose = st.slider("각도 스무딩 (EMA pose, 작을수록 부드러움)", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
            fallback_ttl = st.number_input("얼굴 소실 시 마지막 포즈 유지 프레임 수 (TTL)", min_value=0, max_value=60, value=15)
            
        max_frames = 0
        
    else: # Blur / Pixelate
        blur_style = st.selectbox("블러 방식", ["Gaussian Blur (흐리게)", "Pixelate (모자이크)"])
        blur_radius = st.slider("블러 강도 (Radius)", min_value=1.0, max_value=50.0, value=18.0, step=1.0)
        pixel_size = st.slider("픽셀 크기 (모자이크 크기)", min_value=2, max_value=32, value=8, step=1)
        max_frames = st.number_input("최대 처리 프레임 (0 = 전체)", value=0, min_value=0)
        
    st.divider()
    
    # 3. 비식별화 실행 버튼 (Direct module calls)
    ready = st.session_state.input_path and st.session_state.det_path
    if not ready:
        st.info("왼쪽 화면에서 원본 파일을 업로드하고 얼굴 검출이 완료되면 실행 버튼이 활성화됩니다.")
    else:
        if st.button("🚀 비식별화 파이프라인 구동 시작", type="primary", use_container_width=True):
            inp = st.session_state.input_path
            out = str(WORK / (Path(inp).stem + "_anonymized" + Path(inp).suffix))
            st.session_state.output_path = out
            
            with st.spinner("비식별화 가공 작업이 열심히 실행 중입니다..."):
                try:
                    if method == "블러 / 픽셀화":
                        run_mode = "blur" if blur_style == "Gaussian Blur (흐리게)" else "pixelate"
                        apply_blur_anonymization(
                            input_path=inp,
                            output_path=out,
                            detections=st.session_state.det_path,
                            keep_track_ids=st.session_state.keep_ids,
                            fallback_mode=run_mode,
                            fallback_blur_radius=blur_radius,
                            fallback_pixel_size=pixel_size,
                            max_frames=max_frames
                        )
                    elif method == "스티커 이미지 붙이기":
                        apply_sticker_anonymization(
                            input_path=inp,
                            output_path=out,
                            detections=st.session_state.det_path,
                            keep_track_ids=st.session_state.keep_ids,
                            emoji_char=emoji_char,
                            sticker_png_path=custom_sticker_path,
                            min_face_size=sticker_min_face_size,
                            max_frames=max_frames
                        )
                    elif method == "3D Helmet Overlay (3D 헬멧 씌우기)":
                        apply_3d_anonymization(
                            input_path=inp,
                            output_path=out,
                            detections=st.session_state.det_path,
                            keep_track_ids=st.session_state.keep_ids,
                            helmet_scale=helmet_scale,
                            y_shift=y_shift,
                            z_shift=z_shift,
                            size_deadband=size_deadband,
                            pos_deadband=pos_deadband,
                            ema_size=ema_size,
                            ema_pos=ema_pos,
                            ema_pose=ema_pose,
                            fallback_ttl=fallback_ttl
                        )
                    else: # Generative Diffusion
                        apply_diffusion_anonymization(
                            input_path=inp,
                            output_path=out,
                            detections=st.session_state.det_path,
                            keep_track_ids=st.session_state.keep_ids,
                            fallback_mode="blur",
                            seed=int(seed),
                            mask_mode=mask_mode,
                            inpaint_scope=inpaint_scope,
                            ref_path=st.session_state.get("ref_path"),
                            ref_mode=diff_mode,
                            max_frames=int(max_frames)
                        )
                    st.success("🎉 비식별화 처리가 완료되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 처리 중 에러 발생: {e}")
                    
    # 4. 결과 출력
    out_path = st.session_state.get("output_path")
    if out_path and Path(out_path).exists():
        st.markdown("---")
        st.markdown("### 📦 최종 처리 결과 비교")
        
        is_video = Path(out_path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        
        if is_video:
            st.video(out_path)
            
            if preview:
                res_frame = get_first_frame(out_path)
                if res_frame:
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.image(preview, caption="원본 (첫 프레임)", use_container_width=True)
                    with res_col2:
                        st.image(res_frame, caption="비식별화 결과 (첫 프레임)", use_container_width=True)
        else:
            result_img = Image.open(out_path)
            res_col1, res_col2 = st.columns(2)
            if preview:
                with res_col1:
                    st.image(preview, caption="원본 사진", use_container_width=True)
                with res_col2:
                    st.image(result_img, caption="비식별화 가공 사진", use_container_width=True)
            else:
                st.image(result_img, caption="비식별화 가공 사진", use_container_width=True)
                
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ 결과 파일 내려받기", f.read(),
                file_name=Path(out_path).name,
                mime="video/mp4" if is_video else "image/jpeg",
                use_container_width=True
            )

"""
Face Anonymization Streamlit UI (심플 버전)
워크플로우:
  1. 이미지/영상 + YOLO JSON 업로드
  2. 탐지된 얼굴 crop 격자 표시
  3. 보호할 Face ID 숫자 입력 (1,5,6,7)
  4. 실행 → 결과 비교
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Face Anonymizer", page_icon="🎭", layout="wide")

# ──────────────────────────────────────────────
# 헬퍼 함수
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


def normalize_to_standard_format(raw_data: Any) -> dict:
    if not raw_data:
        return {"frames": []}
        
    normalized_frames = []

    def normalize_face(item: dict) -> dict:
        face = item.copy()
        tid = get_track_id(face)
        if tid is not None:
            face["track_id"] = tid
        bbox = get_bbox(face)
        if bbox is not None:
            face["bbox"] = list(bbox)
        return face

    if isinstance(raw_data, list):
        faces = [normalize_face(item) for item in raw_data if isinstance(item, dict)]
        normalized_frames.append({"frame_index": 0, "faces": faces})

    elif isinstance(raw_data, dict):
        if "frames" in raw_data:
            frames = raw_data["frames"]
            if isinstance(frames, dict):
                for fidx_str, faces_list in frames.items():
                    clean_key = "".join(c for c in fidx_str if c.isdigit())
                    fidx = int(clean_key) if clean_key else 0
                    faces = []
                    if isinstance(faces_list, list):
                        faces = [normalize_face(item) for item in faces_list if isinstance(item, dict)]
                    normalized_frames.append({"frame_index": fidx, "faces": faces})
            elif isinstance(frames, list):
                for idx, frame_item in enumerate(frames):
                    if isinstance(frame_item, dict):
                        fidx = frame_item.get("frame_index", frame_item.get("frame_idx", frame_item.get("frame", idx)))
                        faces_list = frame_item.get("faces", frame_item.get("detections", frame_item.get("objects", frame_item.get("tracks", []))))
                        faces = []
                        if isinstance(faces_list, list):
                            faces = [normalize_face(item) for item in faces_list if isinstance(item, dict)]
                        normalized_frames.append({"frame_index": fidx, "faces": faces})
        else:
            # E.g. {"frame_0": [...], "frame_5": [...]}
            for key, faces_list in raw_data.items():
                clean_key = "".join(c for c in key if c.isdigit())
                fidx = int(clean_key) if clean_key else 0
                faces = []
                if isinstance(faces_list, list):
                    faces = [normalize_face(item) for item in faces_list if isinstance(item, dict)]
                normalized_frames.append({"frame_index": fidx, "faces": faces})

    # Sort frames by frame_index
    normalized_frames.sort(key=lambda x: x["frame_index"])
    return {"frames": normalized_frames}


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



def get_bbox(face: dict) -> tuple[int,int,int,int] | None:
    raw = face.get("bbox") or face.get("box") or face.get("xyxy")
    if raw is None:
        if all(k in face for k in ("x1","y1","x2","y2")):
            return int(face["x1"]), int(face["y1"]), int(face["x2"]), int(face["y2"])
        if all(k in face for k in ("x","y","w","h")):
            x,y,w,h = face["x"],face["y"],face["w"],face["h"]
            return int(x), int(y), int(x+w), int(y+h)
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        a,b,c,d = [float(v) for v in raw]
        return int(a), int(b), int(c), int(d)
    return None


def get_track_id(face: dict) -> str | None:
    for key in ("track_id","trackId","id","track"):
        if key in face and face[key] is not None:
            return str(face[key])
    return None


def crop_face(image: Image.Image, bbox: tuple, pad: float = 0.3) -> Image.Image:
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    x1 = max(0, x1 - int(w*pad)); y1 = max(0, y1 - int(h*pad))
    x2 = min(image.width, x2 + int(w*pad)); y2 = min(image.height, y2 + int(h*pad))
    return image.crop((x1,y1,x2,y2)).resize((150,150), Image.Resampling.LANCZOS)


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
        x1,y1,x2,y2 = bbox
        color = (67,233,123) if tid in keep_ids else (245,87,108)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        label = f"ID {tid} {'KEEP' if tid in keep_ids else 'ANON'}"
        draw.text((x1, max(0, y1-18)), label, fill=color, font=font)
    return out


def get_first_frame(path: str) -> Image.Image | None:
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None

def run_auto_yolo_detection(media_path: str) -> dict:
    from ultralytics import YOLO
    
    # yolo_blur/yolo_11_L/best.pt 가 있다면 사용, 없으면 yolov8n-face.pt (자동 다운로드)
    model_path = Path(__file__).parent / "yolo_blur" / "yolo_11_L" / "best.pt"
    if not model_path.exists():
        model_path = "yolov8n-face.pt"
        
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        return {"frames": []}
        
    frames_data = []
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임이 너무 많으면 속도를 위해 2프레임마다 건너뛸 수도 있지만,
        # 정확한 트래킹을 위해 매 프레임 track 진행.
        results = model.track(frame, persist=True, conf=0.3, verbose=False)
        faces = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = [i+1 for i in range(len(boxes))]
                
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box[:4])
                faces.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2]
                })
        frames_data.append({"frame_index": idx, "faces": faces})
        idx += 1
        
    cap.release()
    return {"frames": frames_data}


def run_pipeline(input_path, det_path, keep_ids, output_path,
                 fallback, seed, max_frames, mask_preview, mask_mode, inpaint_scope,
                 sticker_anonymize=False, ref_path=None, ref_mode="얼굴 합성 (Face Blend)") -> tuple[bool, str]:
    if sticker_anonymize:
        script = Path(__file__).parent / "face-anonymizer" / "sticker_mode" / "anonymize.py"
    else:
        script = Path(__file__).parent / "face-anonymizer" / "anonymize.py"
    cmd = [
        sys.executable, str(script),
        "--input", input_path,
        "--detections", det_path,
        "--output", output_path,
        "--fallback-mode", fallback,
        "--seed", str(seed),
        "--mask-mode", mask_mode,
        "--inpaint-scope", inpaint_scope,
        "--variant", "fp16",
    ]
    if keep_ids:
        cmd += ["--keep-track-ids", ",".join(sorted(keep_ids))]
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    if mask_preview:
        cmd += ["--mask-preview"]
    if sticker_anonymize:
        cmd += ["--sticker-anonymize"]
    if ref_path and os.path.exists(ref_path):
        if ref_mode == "얼굴 합성 (Face Blend)":
            cmd += ["--reference-face-images", ref_path]
        elif ref_mode == "아이덴티티 보존 (InstantID/IP-Adapter)":
            cmd += ["--reference-identity-images", ref_path]
        elif ref_mode == "프롬프트 추출 (Prompt Only)":
            cmd += ["--reference-images", ref_path]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(script.parent) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                           env=env, cwd=str(script.parent))
        return (True, r.stdout) if r.returncode == 0 else (False, r.stderr)
    except subprocess.TimeoutExpired:
        return False, "타임아웃 (60분 초과)"
    except Exception as e:
        return False, str(e)


# ──────────────────────────────────────────────
# 세션 상태
# ──────────────────────────────────────────────
defaults = {
    "input_path": None, "det_path": None,
    "detections": [], "all_track_ids": [], "preview": None,
    "keep_ids": set(), "output_path": None,
    "work_dir": tempfile.mkdtemp(prefix="face_anon_"),
    "ref_path": None,
    "ref_preview": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

WORK = Path(st.session_state.work_dir)

# ──────────────────────────────────────────────
# 타이틀 및 프리미엄 CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Font Override */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Accent Gradient */
    .title-gradient {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem !important;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2.0rem;
    }
    
    /* Custom Card Style for Previews and Sections */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5) !important;
        background: linear-gradient(135deg, #9333ea 0%, #2563eb 100%) !important;
    }
    
    /* File Uploader Custom Aesthetics */
    section[data-testid="stFileUploader"] {
        border: 1px dashed rgba(168, 85, 247, 0.4) !important;
        border-radius: 12px !important;
        background-color: rgba(168, 85, 247, 0.02) !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stFileUploader"]:hover {
        border-color: #a855f7 !important;
        background-color: rgba(168, 85, 247, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown('<div class="title-gradient">🎭 Face Anonymizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">YOLO 탐지 결과 기반 선택적 얼굴 비식별화 파이프라인</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 사이드바: 실행 옵션 & 레퍼런스 모드
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 실행 옵션")
    fallback   = st.selectbox("폴백 방식", ["blur", "pixelate", "none"])
    mask_mode  = st.selectbox("마스크 모드", ["auto", "ellipse", "segmentation", "landmark", "bbox", "sam"])
    inpaint_scope = st.selectbox("Inpaint 범위", ["face-crop", "full-frame"])
    seed       = st.number_input("랜덤 시드", value=1234, min_value=0)
    max_frames = st.number_input("최대 프레임 (영상, 0=전체)", value=24, min_value=0)
    mask_only  = st.checkbox("마스크 미리보기만 (Diffusion 생략)")
    sticker_anonymize = st.checkbox("Fast Sticker 모드 (비식별화 고속화)", value=True, help="얼굴 ID별로 처음 등장할 때만 Diffusion을 수행하고, 이후 프레임은 생성이 아닌 스티커 합성 방식을 적용하여 초고속으로 처리합니다. 동영상 비식별화의 연산 속도를 대폭 줄일 수 있습니다.")
    
    st.divider()
    st.markdown("### 🧬 레퍼런스 설정")
    ref_mode = st.selectbox(
        "레퍼런스 모드",
        ["얼굴 합성 (Face Blend)", "아이덴티티 보존 (InstantID/IP-Adapter)", "프롬프트 추출 (Prompt Only)"],
        help="레퍼런스 사진 업로드 시 사용할 비식별화 방식입니다."
    )

    st.divider()
    if st.button("🔄 전체 초기화"):
        for k in ["input_path","det_path","detections","all_track_ids","preview","keep_ids","output_path","ref_path","ref_preview"]:
            st.session_state[k] = defaults.get(k)
        st.rerun()

# ──────────────────────────────────────────────
# 메인 레이아웃: 2컬럼 구성
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 원본 및 레퍼런스 업로드")
    
    # 1. 원본 미디어 업로드
    media_file = st.file_uploader(
        "원본 이미지 / 영상 파일 선택",
        type=["jpg","jpeg","png","bmp","webp","mp4","mov","avi","mkv"],
        key="media_file_uploader"
    )
    
    if media_file:
        p = WORK / media_file.name
        p.write_bytes(media_file.getbuffer())
        st.session_state.input_path = str(p)
        ext = p.suffix.lower()
        if ext in {".mp4",".mov",".avi",".mkv",".m4v"}:
            st.session_state.preview = get_first_frame(str(p))
            st.success(f"🎬 영상 파일 등록 완료: {media_file.name}")
            # 영상 업로드 시 바로 보이기
            st.video(str(p))
        else:
            st.session_state.preview = Image.open(p).convert("RGB")
            st.success(f"🖼️ 이미지 파일 등록 완료: {media_file.name}")
            # 이미지 업로드 시 바로 보이기
            st.image(st.session_state.preview, use_container_width=True)

    st.divider()
    
    # 2. 레퍼런스 사진 업로드 (원본 아래 배치)
    ref_file = st.file_uploader(
        "레퍼런스 사진 업로드 (선택)",
        type=["jpg","jpeg","png","bmp","webp"],
        key="ref_file_uploader"
    )
    
    if ref_file:
        p = WORK / ref_file.name
        p.write_bytes(ref_file.getbuffer())
        st.session_state.ref_path = str(p)
        st.session_state.ref_preview = Image.open(p).convert("RGB")
        st.success(f"🧬 레퍼런스 사진 등록 완료: {ref_file.name}")
        # 레퍼런스 사진 업로드 시 바로 보이기
        st.image(st.session_state.ref_preview, caption="등록된 레퍼런스 사진", use_container_width=True)

    st.divider()
    
    # 3. YOLO JSON 업로드 (삭제됨 - 자동 탐지로 대체)
    # 자동 YOLO 탐지 수행 여부 확인
    if st.session_state.input_path and not st.session_state.det_path:
        with st.spinner("🤖 YOLO 모델을 통해 얼굴을 자동 탐지하고 있습니다..."):
            try:
                normalized = run_auto_yolo_detection(st.session_state.input_path)
                # Save to JSON
                p = WORK / "auto_detections.json"
                p.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
                st.session_state.det_path = str(p)
                
                dets = parse_detections(normalized)
                st.session_state.detections = dets
                st.session_state.all_track_ids = extract_all_track_ids(normalized)
                st.success(f"✅ 얼굴 자동 탐지 완료 (고유 ID {len(st.session_state.all_track_ids)}개 발견)")
                st.rerun()
            except Exception as e:
                st.error(f"YOLO 탐지 중 오류 발생: {e}")


    # 4. 탐지된 얼굴 크롭 격자 및 ID 선택
    dets = st.session_state.detections
    preview = st.session_state.preview
    
    if dets and preview:
        st.divider()
        st.markdown("### 🔍 탐지된 얼굴 분석")
        
        # Crop 갤러리
        COLS = 5
        rows = [dets[i:i+COLS] for i in range(0, len(dets), COLS)]
        for row in rows:
            cols = st.columns(len(row))
            for col, face in zip(cols, row):
                tid = get_track_id(face)
                bbox = get_bbox(face)
                with col:
                    if bbox:
                        try:
                            img = crop_face(preview, bbox)
                            st.image(img, use_container_width=True)
                        except Exception:
                            pass
                        protected = tid in st.session_state.keep_ids
                        if protected:
                            st.markdown(f"<p style='text-align:center;color:#2ecc71;font-weight:bold;margin:2px;font-size:12px'>✅ ID {tid}</p>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p style='text-align:center;color:#e74c3c;font-weight:bold;margin:2px;font-size:12px'>🎭 ID {tid}</p>", unsafe_allow_html=True)

        st.markdown("#### 🛡️ 보호할 얼굴 ID 선택")
        all_ids_list = st.session_state.get("all_track_ids") or [get_track_id(f) for f in dets if get_track_id(f)]
        all_ids = ", ".join(all_ids_list)
        st.caption(f"탐지된 모든 Face ID: **{all_ids}**")
        st.caption("비식별화하지 않고 그대로 유지할(보호할) ID를 쉼표로 구분해 입력하세요.")

        keep_input = st.text_input(
            "보호할 ID 입력란",
            placeholder="예: 1,5,6",
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("✅ ID 적용", use_container_width=True):
                st.session_state.keep_ids = {x.strip() for x in keep_input.split(",") if x.strip()}
                st.rerun()
        with c2:
            if st.button("🔄 전체 비식별화", use_container_width=True):
                st.session_state.keep_ids = set()
                st.rerun()

        keep_ids = st.session_state.keep_ids
        all_set  = set(all_ids_list)
        anon_set = all_set - keep_ids

        # 간단 매트릭
        m1, m2, m3 = st.columns(3)
        m1.metric("전체 얼굴", len(all_set))
        m2.metric("보호 (Keep)", len(keep_ids))
        m3.metric("비식별화 (Anon)", len(anon_set))

        # BBox 미리보기
        if keep_ids or anon_set:
            st.image(draw_bboxes(preview, dets, keep_ids),
                     caption="탐지 및 상태 시각화 (초록=보호, 빨강=비식별화)", use_container_width=True)

with col_right:
    st.markdown("### 🎯 비식별화 실행 및 결과")
    
    ready = st.session_state.input_path and st.session_state.det_path
    
    if not ready:
        st.info("왼쪽 영역에서 이미지/영상과 YOLO JSON 결과를 모두 등록하면 비식별화 실행 준비가 완료됩니다.")
    else:
        # 실행 버튼
        if st.button("🚀 비식별화 실행", type="primary", use_container_width=True):
            inp  = st.session_state.input_path
            out  = str(WORK / (Path(inp).stem + "_anonymized" + Path(inp).suffix))
            st.session_state.output_path = out

            with st.spinner("비식별화 작업이 진행 중입니다... (Generative AI 작동으로 수 분이 걸릴 수 있습니다)"):
                ok, log = run_pipeline(
                    inp, st.session_state.det_path,
                    st.session_state.keep_ids, out,
                    fallback, int(seed), int(max_frames),
                    mask_only, mask_mode, inpaint_scope,
                    sticker_anonymize=sticker_anonymize,
                    ref_path=st.session_state.get("ref_path"),
                    ref_mode=ref_mode
                )

            if ok:
                st.success("🎉 비식별화 처리가 성공적으로 완료되었습니다!")
                st.rerun()
            else:
                st.error("❌ 비식별화 처리 도중 오류가 발생했습니다.")
                st.code(log)

    # 결과 디스플레이
    out_path = st.session_state.get("output_path")
    if out_path and Path(out_path).exists():
        st.markdown("---")
        st.markdown("#### 📦 최종 결과 비교")

        is_video = Path(out_path).suffix.lower() in {".mp4",".mov",".avi",".mkv"}

        if is_video:
            # 비디오 출력
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
            # 이미지 출력 및 비교
            result_img = Image.open(out_path)
            res_col1, res_col2 = st.columns(2)
            if preview:
                with res_col1:
                    st.image(preview, caption="원본 이미지", use_container_width=True)
                with res_col2:
                    st.image(result_img, caption="비식별화 결과", use_container_width=True)
            else:
                st.image(result_img, caption="비식별화 결과", use_container_width=True)

        # 다운로드 버튼
        with open(out_path, "rb") as f:
            st.download_button("⬇️ 비식별화 결과 파일 다운로드", f.read(),
                               file_name=Path(out_path).name,
                               mime="video/mp4" if is_video else "image/jpeg",
                               use_container_width=True)


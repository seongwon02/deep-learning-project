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


def run_pipeline(input_path, det_path, keep_ids, output_path,
                 fallback, seed, max_frames, mask_preview, mask_mode, inpaint_scope) -> tuple[bool, str]:
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
    ]
    if keep_ids:
        cmd += ["--keep-track-ids", ",".join(sorted(keep_ids))]
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    if mask_preview:
        cmd += ["--mask-preview"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(script.parent) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env=env, cwd=str(script.parent))
        return (True, r.stdout) if r.returncode == 0 else (False, r.stderr)
    except subprocess.TimeoutExpired:
        return False, "타임아웃 (10분 초과)"
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

WORK = Path(st.session_state.work_dir)

# ──────────────────────────────────────────────
# 타이틀
# ──────────────────────────────────────────────
st.title("🎭 Face Anonymizer")
st.caption("YOLO 탐지 결과 기반 선택적 얼굴 비식별화 파이프라인")
st.divider()

# ──────────────────────────────────────────────
# 사이드바: 실행 옵션만
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("실행 옵션")
    fallback   = st.selectbox("폴백 방식", ["blur", "pixelate", "none"])
    mask_mode  = st.selectbox("마스크 모드", ["auto", "ellipse", "segmentation", "landmark", "bbox", "sam"])
    inpaint_scope = st.selectbox("Inpaint 범위", ["face-crop", "full-frame"])
    seed       = st.number_input("랜덤 시드", value=1234, min_value=0)
    max_frames = st.number_input("최대 프레임 (영상, 0=전체)", value=24, min_value=0)
    mask_only  = st.checkbox("마스크 미리보기만 (Diffusion 생략)")

    st.divider()
    if st.button("초기화"):
        for k in ["input_path","det_path","detections","all_track_ids","preview","keep_ids","output_path"]:
            st.session_state[k] = defaults.get(k)
        st.rerun()

# ──────────────────────────────────────────────
# STEP 1: 파일 업로드
# ──────────────────────────────────────────────
st.subheader("① 파일 업로드")
col1, col2 = st.columns(2)

with col1:
    media_file = st.file_uploader(
        "이미지 / 영상",
        type=["jpg","jpeg","png","bmp","webp","mp4","mov","avi","mkv"],
    )
with col2:
    json_file = st.file_uploader("YOLO 탐지 결과 JSON", type=["json"])

if media_file:
    p = WORK / media_file.name
    p.write_bytes(media_file.getbuffer())
    st.session_state.input_path = str(p)
    ext = p.suffix.lower()
    if ext in {".mp4",".mov",".avi",".mkv",".m4v"}:
        st.session_state.preview = get_first_frame(str(p))
        st.info(f"🎬 영상 업로드: {media_file.name}")
    else:
        st.session_state.preview = Image.open(p).convert("RGB")
        st.info(f"🖼️ 이미지 업로드: {media_file.name}")

if json_file:
    p = WORK / json_file.name
    p.write_bytes(json_file.getbuffer())
    st.session_state.det_path = str(p)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        normalized = normalize_to_standard_format(raw)
        p.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
        
        dets = parse_detections(normalized)
        st.session_state.detections = dets
        st.session_state.all_track_ids = extract_all_track_ids(normalized)
        st.success(f"✅ JSON 로드 완료 및 표준 변환 완료 — 전체 {len(st.session_state.all_track_ids)}개 고유 ID 탐지됨")
    except Exception as e:
        st.error(f"JSON 파싱 오류: {e}")

# ──────────────────────────────────────────────
# STEP 2: 얼굴 Crop 갤러리
# ──────────────────────────────────────────────
dets = st.session_state.detections
preview = st.session_state.preview

if dets and preview:
    st.divider()
    st.subheader("② 탐지된 얼굴")

    COLS = 6
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
                        st.markdown(f"<p style='text-align:center;color:#2ecc71;font-weight:bold;margin:2px'>✅ ID {tid}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='text-align:center;color:#e74c3c;font-weight:bold;margin:2px'>🎭 ID {tid}</p>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# STEP 3: ID 입력
# ──────────────────────────────────────────────
if dets and preview:
    st.divider()
    st.subheader("③ 보호할 얼굴 ID 선택")

    all_ids_list = st.session_state.get("all_track_ids") or [get_track_id(f) for f in dets if get_track_id(f)]
    all_ids = ", ".join(all_ids_list)
    st.caption(f"탐지된 모든 Face ID: **{all_ids}**")
    st.caption("비식별화 **하지 않을** 얼굴의 ID를 입력하세요. 나머지는 전부 비식별화됩니다.")

    keep_input = st.text_input(
        "보호할 ID (쉼표 구분)",
        placeholder="예: 1,5,6,7",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("✅ 적용", use_container_width=True):
            st.session_state.keep_ids = {x.strip() for x in keep_input.split(",") if x.strip()}
            st.rerun()
    with c2:
        if st.button("🔄 초기화 (전부 비식별화)", use_container_width=True):
            st.session_state.keep_ids = set()
            st.rerun()

    keep_ids = st.session_state.keep_ids
    all_set  = set(all_ids_list)
    anon_set = all_set - keep_ids

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 얼굴", len(all_set))
    c2.metric("보호 (Keep)", len(keep_ids))
    c3.metric("비식별화 (Anon)", len(anon_set))

    # BBox 미리보기
    if keep_ids or anon_set:
        st.image(draw_bboxes(preview, dets, keep_ids),
                 caption="초록=보호  |  빨강=비식별화", use_container_width=True)

# ──────────────────────────────────────────────
# STEP 4: 실행
# ──────────────────────────────────────────────
if dets and preview:
    st.divider()
    st.subheader("④ 실행")

    ready = st.session_state.input_path and st.session_state.det_path
    if not ready:
        st.warning("이미지/영상과 JSON을 먼저 업로드하세요.")
    else:
        if st.button("🚀 비식별화 실행", type="primary", use_container_width=True):
            inp  = st.session_state.input_path
            out  = str(WORK / (Path(inp).stem + "_anonymized" + Path(inp).suffix))
            st.session_state.output_path = out

            with st.spinner("실행 중... (Diffusion은 수 분 소요됩니다)"):
                ok, log = run_pipeline(
                    inp, st.session_state.det_path,
                    st.session_state.keep_ids, out,
                    fallback, int(seed), int(max_frames),
                    mask_only, mask_mode, inpaint_scope,
                )

            if ok:
                st.success("✅ 완료!")
                st.rerun()
            else:
                st.error("실패")
                st.code(log[:2000])

# ──────────────────────────────────────────────
# STEP 5: 결과
# ──────────────────────────────────────────────
out_path = st.session_state.get("output_path")
if out_path and Path(out_path).exists():
    st.divider()
    st.subheader("⑤ 결과")

    is_video = Path(out_path).suffix.lower() in {".mp4",".mov",".avi",".mkv"}

    if is_video:
        st.video(out_path)
        if preview:
            res_frame = get_first_frame(out_path)
            if res_frame:
                a, b = st.columns(2)
                a.image(preview,   caption="원본 (첫 프레임)", use_container_width=True)
                b.image(res_frame, caption="결과 (첫 프레임)", use_container_width=True)
    else:
        result_img = Image.open(out_path)
        a, b, c = st.columns(3)
        if preview:
            a.image(preview,    caption="① 원본",        use_container_width=True)
            a.image(draw_bboxes(preview, dets, st.session_state.keep_ids),
                                caption="② YOLO BBox",  use_container_width=True)
        c.image(result_img, caption="③ 최종 결과",   use_container_width=True)

    with open(out_path, "rb") as f:
        st.download_button("⬇️ 결과 다운로드", f.read(),
                           file_name=Path(out_path).name,
                           mime="video/mp4" if is_video else "image/jpeg")

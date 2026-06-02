"""
3D Mesh (Helmet / Masks) video overlay anonymizer processor.
Provides clean APIs to render 3D objects mapped to detected face positions and poses.
"""

import cv2
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Optional
from insightface.app import FaceAnalysis

def load_obj(path):
    """Load Wavefront OBJ file. Returns (vertices, faces) where faces = [(tri_indices, material), ...]"""
    vertices = []
    faces = []
    current_material = "body"
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("usemtl "):
                mat = line.split()[1]
                if "glass" in mat.lower():
                    current_material = "glass"
                else:
                    current_material = "body"
            elif line.startswith("f "):
                parts = line.split()
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split("/")[0])
                    if idx < 0:
                        idx = len(vertices) + idx + 1
                    face_verts.append(idx - 1)
                if len(face_verts) >= 3:
                    for i in range(1, len(face_verts) - 1):
                        faces.append(([face_verts[0], face_verts[i], face_verts[i+1]], current_material))
    return np.array(vertices, dtype=np.float32), faces

def get_rotation_matrix(pitch, yaw, roll):
    p = math.radians(pitch)
    y = math.radians(yaw)
    r = math.radians(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(p), -math.sin(p)],
        [0, math.sin(p), math.cos(p)]
    ])
    Ry = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ])
    Rz = np.array([
        [math.cos(r), -math.sin(r), 0],
        [math.sin(r), math.cos(r), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if float(boxAArea + boxBArea - interArea) <= 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def get_bbox(face: dict) -> tuple[int, int, int, int] | None:
    raw = face.get("bbox") or face.get("box") or face.get("xyxy")
    if raw is None:
        return None
    return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])

def get_track_id(face: dict) -> str | None:
    for key in ("track_id", "trackId", "id", "track"):
        if key in face and face[key] is not None:
            return str(face[key])
    return None

def load_detections(det_path: Path) -> dict[int, list]:
    with open(det_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_frame = {}
    if isinstance(data, dict) and "frames" in data:
        for item in data["frames"]:
            fidx = item.get("frame_index", 0)
            faces = item.get("faces", [])
            by_frame[fidx] = faces
    return by_frame

def render_helmet_on_frame(frame, cx, cy, size, pitch, yaw, roll, 
                            vertices, faces, helmet_scale, width, height, light_dir):
    """Render the 3D helmet on a single face in the frame."""
    R = get_rotation_matrix(-pitch, yaw, -roll)
    rotated_vertices = vertices @ R.T
    
    scale = (size * helmet_scale) / 40.0
    
    face_depths = []
    for i, (f_verts, mat) in enumerate(faces):
        z_coords = [rotated_vertices[v_idx][2] for v_idx in f_verts]
        avg_z = sum(z_coords) / 3.0
        face_depths.append((i, avg_z))
        
    face_depths.sort(key=lambda x: x[1])
    
    for f_idx, _ in face_depths:
        f_verts, mat = faces[f_idx]
        
        proj_pts = []
        valid = True
        for v_idx in f_verts:
            rv = rotated_vertices[v_idx]
            px = int(cx + scale * rv[0])
            py = int(cy - scale * rv[1]) # flip Y for screen coordinates
            
            if not (-1000 < px < width + 1000 and -1000 < py < height + 1000):
                valid = False
                break
            proj_pts.append([px, py])
            
        if not valid:
            continue
            
        p1, p2, p3 = rotated_vertices[f_verts[0]], rotated_vertices[f_verts[1]], rotated_vertices[f_verts[2]]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm
        
        intensity = np.dot(normal, light_dir)
        intensity = max(0.1, intensity)
        
        if mat == "glass":
            color_val = int(20 + 80 * intensity)
            color = (color_val, color_val, color_val)
        else:
            color_val = int(50 + 200 * intensity)
            color = (color_val, color_val, color_val)
            
        pts_arr = np.array(proj_pts, dtype=np.int32)
        cv2.fillPoly(frame, [pts_arr], color)

def apply_3d_anonymization(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    detections: Union[Dict[str, Any], Path, str],
    keep_track_ids: Union[Set[str], List[str], str] = set(),
    obj_path: Optional[Union[str, Path]] = None,
    helmet_scale: float = 1.35,
    y_shift: float = -0.15,
    z_shift: float = -6.0,
    size_deadband: float = 0.08,
    pos_deadband: float = 0.04,
    ema_size: float = 0.3,
    ema_pos: float = 0.3,
    ema_pose: float = 0.25,
    fallback_ttl: int = 15
) -> None:
    """
    Applies 3D helmet overlay anonymization to input video.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process keep track IDs
    if isinstance(keep_track_ids, str):
        keep_ids = {x.strip() for x in keep_track_ids.split(",") if x.strip()}
    else:
        keep_ids = set(str(x) for x in keep_track_ids)
        
    # Auto-resolve OBJ path
    curr_dir = Path(__file__).parent.resolve()
    if obj_path is None or str(obj_path) == "":
        obj_path = curr_dir / "assets" / "10517_Motorcycle_Helmet_v01_L3.obj"
    else:
        obj_path = Path(obj_path)
        if not obj_path.exists():
            local_obj = curr_dir / "assets" / obj_path.name
            if local_obj.exists():
                obj_path = local_obj
                
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ model file not found at: {obj_path}")
        
    print(f"[3D Anonymizer] Loading 3D model: {obj_path}")
    vertices, faces = load_obj(obj_path)
    print(f"[3D Anonymizer] Loaded {len(vertices)} vertices and {len(faces)} faces.")
    
    # Center and project to standard coordinates
    center = vertices.mean(axis=0)
    vertices = vertices - center
    
    x_std = -vertices[:, 1]
    y_std = vertices[:, 2]
    z_std = -vertices[:, 0] + z_shift
    vertices = np.stack([x_std, y_std, z_std], axis=1)
    
    # Load detections
    if isinstance(detections, (str, Path)):
        by_frame = load_detections(Path(detections))
    else:
        # Dictionary format
        by_frame = {}
        if isinstance(detections, dict) and "frames" in detections:
            for item in detections["frames"]:
                fidx = item.get("frame_index", 0)
                faces = item.get("faces", [])
                by_frame[fidx] = faces
        else:
            by_frame = detections
            
    # Auto-detect face analyzer models folder
    possible_models_dirs = [
        curr_dir.parent,                                   # real/anon_project/
        curr_dir.parent.parent / "real",                   # real/
        curr_dir.parent.parent                             # project root /
    ]
    models_dir = None
    for p in possible_models_dirs:
        if (p / "models").exists():
            models_dir = p
            break
    if models_dir is None:
        models_dir = curr_dir.parent.parent
        
    print(f"[3D Anonymizer] Initializing InsightFace buffalo_l app using root: {models_dir}")
    app = FaceAnalysis(name="buffalo_l", root=str(models_dir))
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not create output video writer: {output_path}")
        
    light_dir = np.array([0.3, 0.3, 0.9], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    
    smoothed_faces = {}
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="[3D Anonymizer] Rendering 3D Helmet")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            yolo_faces = by_frame.get(frame_idx, [])
            insight_faces = app.get(frame)
            
            matched_yolo_tids = set()
            
            # Phase 1: Process InsightFace detections (pose source)
            for iface in insight_faces:
                ibox = iface.bbox.astype(int)
                
                best_tid = None
                best_iou = 0.0
                for yface in yolo_faces:
                    ybox = get_bbox(yface)
                    if ybox:
                        iou = compute_iou(ibox, ybox)
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = get_track_id(yface)
                            
                if best_tid in keep_ids:
                    if best_tid:
                        matched_yolo_tids.add(best_tid)
                    continue
                    
                if best_tid:
                    matched_yolo_tids.add(best_tid)
                    
                bx1, by1, bx2, by2 = ibox
                bw, bh = bx2 - bx1, by2 - by1
                size = max(bw, bh)
                
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0 + bh * y_shift
                
                pose = iface.pose
                pitch, yaw, roll = pose[0], pose[1], pose[2]
                
                # Apply smoothing
                if best_tid is not None:
                    if best_tid in smoothed_faces:
                        prev = smoothed_faces[best_tid]
                        
                        size_change = abs(size - prev["size"]) / prev["size"]
                        if size_change < size_deadband:
                            size = prev["size"]
                        else:
                            size = ema_size * size + (1 - ema_size) * prev["size"]
                            
                        center_dist = math.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                        if center_dist < pos_deadband * size:
                            cx = prev["cx"]
                            cy = prev["cy"]
                        else:
                            cx = ema_pos * cx + (1 - ema_pos) * prev["cx"]
                            cy = ema_pos * cy + (1 - ema_pos) * prev["cy"]
                            
                        pitch = ema_pose * pitch + (1 - ema_pose) * prev["pose"][0]
                        yaw = ema_pose * yaw + (1 - ema_pose) * prev["pose"][1]
                        roll = ema_pose * roll + (1 - ema_pose) * prev["pose"][2]
                        
                    smoothed_faces[best_tid] = {
                        "cx": cx,
                        "cy": cy,
                        "size": size,
                        "pose": [pitch, yaw, roll],
                        "missed_frames": 0
                    }
                    
                render_helmet_on_frame(
                    frame, cx, cy, size, pitch, yaw, roll,
                    vertices, faces, helmet_scale, width, height, light_dir
                )
                
            # Phase 2: Handle partial occlusion fallback
            for yface in yolo_faces:
                tid = get_track_id(yface)
                if tid is None or tid in matched_yolo_tids or tid in keep_ids:
                    continue
                    
                if tid not in smoothed_faces:
                    continue
                    
                prev = smoothed_faces[tid]
                missed = prev.get("missed_frames", 0) + 1
                
                if missed > fallback_ttl:
                    del smoothed_faces[tid]
                    continue
                    
                ybox = get_bbox(yface)
                if ybox is None:
                    continue
                    
                bx1, by1, bx2, by2 = ybox
                bw, bh = bx2 - bx1, by2 - by1
                size = max(bw, bh)
                
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0 + bh * y_shift
                
                size_change = abs(size - prev["size"]) / prev["size"]
                if size_change < size_deadband:
                    size = prev["size"]
                else:
                    size = ema_size * size + (1 - ema_size) * prev["size"]
                    
                center_dist = math.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                if center_dist < pos_deadband * size:
                    cx = prev["cx"]
                    cy = prev["cy"]
                else:
                    cx = ema_pos * cx + (1 - ema_pos) * prev["cx"]
                    cy = ema_pos * cy + (1 - ema_pos) * prev["cy"]
                    
                pitch, yaw, roll = prev["pose"]
                
                smoothed_faces[tid] = {
                    "cx": cx,
                    "cy": cy,
                    "size": size,
                    "pose": [pitch, yaw, roll],
                    "missed_frames": missed
                }
                
                render_helmet_on_frame(
                    frame, cx, cy, size, pitch, yaw, roll,
                    vertices, faces, helmet_scale, width, height, light_dir
                )
                
            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()
    print(f"[3D Anonymizer] Completed. Saved result to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Helmet Mesh CLI Wrapper")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--obj", type=Path, default=None)
    parser.add_argument("--keep-track-ids", type=str, default="")
    parser.add_argument("--helmet-scale", type=float, default=1.35)
    parser.add_argument("--y-shift", type=float, default=-0.15)
    parser.add_argument("--z-shift", type=float, default=-6.0)
    parser.add_argument("--size-deadband", type=float, default=0.08)
    parser.add_argument("--pos-deadband", type=float, default=0.04)
    parser.add_argument("--ema-size", type=float, default=0.3)
    parser.add_argument("--ema-pos", type=float, default=0.3)
    parser.add_argument("--ema-pose", type=float, default=0.25)
    parser.add_argument("--fallback-ttl", type=int, default=15)
    args = parser.parse_args()
    
    apply_3d_anonymization(
        args.input, args.output, args.detections,
        keep_track_ids=args.keep_track_ids,
        obj_path=args.obj,
        helmet_scale=args.helmet_scale,
        y_shift=args.y_shift,
        z_shift=args.z_shift,
        size_deadband=args.size_deadband,
        pos_deadband=args.pos_deadband,
        ema_size=args.ema_size,
        ema_pos=args.ema_pos,
        ema_pose=args.ema_pose,
        fallback_ttl=args.fallback_ttl
    )

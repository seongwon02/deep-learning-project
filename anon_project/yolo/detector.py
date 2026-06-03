"""
Face detection and tracking module using YOLOv8/v11.
Provides clean APIs to track faces in images or videos and extract bounding boxes and landmarks.
"""

import cv2
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from ultralytics import YOLO

def run_tracking(
    input_path: str | Path,
    model_path: Optional[str | Path] = None,
    output_json_path: Optional[str | Path] = None,
    conf_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Tracks faces in a video or detects faces in a single image.
    Returns a dictionary structure: {"frames": [{"frame_index": i, "faces": [{"track_id": id, "bbox": [x1, y1, x2, y2], "keypoints": [...]}]}]}
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
    # Resolve YOLO model path
    if model_path is None:
        # Search relative to workspace or project root
        curr_dir = Path(__file__).parent.resolve()
        possible_paths = [
            curr_dir.parent / "face_yolov8l.pt",
            curr_dir.parent / "yolov8l-face.pt",
            curr_dir.parent / "face_yolov8n.pt",
            curr_dir.parent.parent / "real" / "face_yolov8l.pt",
            curr_dir.parent.parent / "real" / "yolov8l-face.pt",
            curr_dir.parent.parent / "real" / "face_yolov8n.pt",
            curr_dir.parent.parent / "face_yolov8l.pt",
            curr_dir.parent.parent / "yolov8l-face.pt",
            curr_dir.parent.parent / "face_yolov8n.pt",
            curr_dir.parent.parent / "yolo11l.pt",
            "yolov8n.pt"
        ]
        for p in possible_paths:
            if isinstance(p, Path) and p.exists():
                model_path = p
                break
        if model_path is None:
            model_path = "yolov8n.pt"

    print(f"[YOLO Tracker] Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    ext = input_path.suffix.lower()
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    
    frames_data = []
    
    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[YOLO Tracker] Video Info: {width}x{height}, Total Frames: {total_frames}")
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)
            faces = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # Extract track IDs or assign indexes if track id is missing
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    track_ids = [i + 1 for i in range(len(boxes))]
                    
                # Extract landmarks (keypoints) if present
                kps_list = []
                if results[0].keypoints is not None:
                    kps_list = results[0].keypoints.xy.cpu().numpy()
                    
                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Bound checks
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    face_data = {
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2]
                    }
                    if i < len(kps_list):
                        face_data["keypoints"] = [[float(pt[0]), float(pt[1])] for pt in kps_list[i]]
                    else:
                        face_data["keypoints"] = []
                    faces.append(face_data)
                    
            frames_data.append({"frame_index": idx, "faces": faces})
            idx += 1
            if idx % 100 == 0:
                print(f"[YOLO Tracker] Processed {idx}/{total_frames} frames")
                
        cap.release()
    else:
        # Single Image
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise IOError(f"Could not read image: {input_path}")
        height, width = frame.shape[:2]
        
        # Non-persist tracking for single frames
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        faces = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = [i + 1 for i in range(len(boxes))]
            
            kps_list = []
            if results[0].keypoints is not None:
                kps_list = results[0].keypoints.xy.cpu().numpy()
                
            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box[:4])
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                face_data = {
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2]
                }
                if i < len(kps_list):
                    face_data["keypoints"] = [[float(pt[0]), float(pt[1])] for pt in kps_list[i]]
                else:
                    face_data["keypoints"] = []
                faces.append(face_data)
                
        frames_data.append({"frame_index": 0, "faces": faces})
        
    result_dict = {"frames": frames_data}
    
    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print(f"[YOLO Tracker] Saved detections to: {output_json_path}")
        
    return result_dict

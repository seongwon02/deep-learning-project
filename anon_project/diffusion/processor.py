"""
Stable Diffusion XL and InstantID face anonymizer processor.
Provides a clean Python API to interact with the diffusion anonymization pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Optional

# Set up paths to properly import the local diffusion packages
curr_dir = Path(__file__).parent.resolve()
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

def apply_diffusion_anonymization(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    detections: Union[Dict[str, Any], Path, str],
    keep_track_ids: Union[Set[str], List[str], str] = set(),
    fallback_mode: str = "blur",
    seed: int = 1234,
    mask_mode: str = "sam",
    inpaint_scope: str = "face-crop",
    ref_path: Optional[Union[str, Path]] = None,
    ref_mode: str = "Face Blend",
    max_frames: int = 0
) -> None:
    """
    Main API to execute Diffusion / InstantID face anonymization.
    It wraps the large anonymize.py pipeline inside a clean in-memory call by mocking sys.argv.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Resolve detections path
    # If detections is a dict, we write it to a temporary JSON so that the pipeline can parse it
    temp_det_json = None
    if isinstance(detections, (str, Path)):
        det_path = Path(detections)
    else:
        import tempfile
        import json
        temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
        json.dump(detections, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        temp_det_json = temp_file.name
        det_path = Path(temp_det_json)
        
    # Standardize keep IDs
    if isinstance(keep_track_ids, str):
        keep_ids_str = keep_track_ids
    else:
        keep_ids_str = ",".join(str(x) for x in keep_track_ids)
        
    # Build CLI arguments to pass to the mocked sys.argv
    args_list = [
        "anonymize.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--detections", str(det_path),
        "--fallback-mode", fallback_mode,
        "--seed", str(seed),
        "--mask-mode", mask_mode,
        "--inpaint-scope", inpaint_scope,
        "--variant", "fp16" # default recommended variant
    ]
    
    if keep_ids_str:
        args_list += ["--keep-track-ids", keep_ids_str]
        
    if max_frames > 0:
        args_list += ["--max-frames", str(max_frames)]
        
    if ref_path and Path(ref_path).exists():
        ref_path = str(Path(ref_path).resolve())
        if ref_mode == "얼굴 합성 (Face Blend)":
            args_list += ["--reference-face-images", ref_path]
        elif ref_mode == "아이덴티티 보존 (InstantID/IP-Adapter)":
            args_list += ["--reference-identity-images", ref_path]
        elif ref_mode == "프롬프트 추출 (Prompt Only)":
            args_list += ["--reference-images", ref_path]
            
    print(f"[Diffusion Anonymizer] Running in-memory pipeline with args: {args_list[1:]}")
    
    # Store original sys.argv and sys.path
    orig_argv = sys.argv
    orig_path = sys.path
    
    # Mock sys.argv
    sys.argv = args_list
    # Inject current directory so it can resolve local modules (like style_presets)
    sys.path = [str(curr_dir)] + sys.path
    
    try:
        # Import and run the main function from anonymize.py
        from anonymize import main as diffusion_main
        diffusion_main()
    except Exception as e:
        print(f"[Diffusion Anonymizer] Error during run: {e}")
        raise e
    finally:
        # Restore sys.argv and sys.path
        sys.argv = orig_argv
        sys.path = orig_path
        
        # Cleanup temporary JSON if created
        if temp_det_json and Path(temp_det_json).exists():
            try:
                Path(temp_det_json).unlink()
            except Exception:
                pass
                
    print(f"[Diffusion Anonymizer] Process completed. Saved to: {output_path}")

if __name__ == "__main__":
    # If run directly as a CLI script, forward to anonymize.py
    import subprocess
    cmd = [sys.executable, str(curr_dir / "anonymize.py")] + sys.argv[1:]
    subprocess.run(cmd)

# Face Anonymizer Prototype

SDXL inpainting face anonymization prototype for short image/video demos.

The YOLO/tracking side should pass `track_id` and face `bbox` data. This tool turns selected tracks into a soft face mask, keeps chosen IDs untouched, and inpaints only the masked face regions.

Boundary:

```text
YOLO/tracker: detect faces, assign track_id, write JSON
SDXL tool: select tracks, build masks/crops, assign synthetic identity, run inpainting, report fallbacks
```

## Recommended Checkpoint

Default:

```text
OzzyGT/RealVisXL_V4.0_inpainting
```

Why this one: it is an SDXL inpainting checkpoint tuned for photorealistic RealVisXL outputs and is available as a Diffusers-compatible Hugging Face model with `openrail++` license metadata.

Fallback:

```text
diffusers/stable-diffusion-xl-1.0-inpainting-0.1
```

Use the fallback when you want the most standard SDXL inpainting baseline.

## Install

Use a virtual environment. The first real generation run downloads the SDXL checkpoint into the Hugging Face cache.

```bash
python3 -m venv .venv-face-anon
source .venv-face-anon/bin/activate
pip install -r face-anonymizer/requirements.txt
```

On Apple Silicon, PyTorch usually uses `mps`. On NVIDIA, it uses `cuda`.

## Detection JSON Contract

Preferred format:

```json
{
  "frames": [
    {
      "frame_index": 0,
      "faces": [
        {
          "track_id": 1,
          "bbox": [128, 72, 248, 226],
          "segmentation": [[128, 100, 150, 76, 222, 78, 248, 104, 236, 214, 188, 226, 140, 214]],
          "confidence": 0.98
        },
        {
          "track_id": 2,
          "bbox": [420, 84, 536, 232],
          "synthetic_id": "synthetic_male_01",
          "confidence": 0.96
        }
      ]
    }
  ]
}
```

`bbox` defaults to `xyxy`: `[x1, y1, x2, y2]`. `xywh` is also supported with `--bbox-format xywh`.

Mask priority with `--mask-mode auto`:

```text
segmentation polygon -> landmark convex hull -> bbox ellipse fallback
```

For character or animal replacement, the default `auto` mask mode is changed to `bbox` after the reference routes to IP-Adapter. This keeps generation bounded to the YOLO box instead of asking another model to infer the face region. If you explicitly want SAM, use `--mask-mode sam`; the YOLO-side `bbox` is passed to SAM as a box prompt, and the resulting mask is still dilated/blurred by the normal mask knobs:

```bash
python face-anonymizer/anonymize.py \
  --input input.jpg \
  --detections detections.json \
  --mask-mode sam \
  --mask-expansion 1.2 \
  --sam-model-id facebook/sam-vit-base \
  --output local/anonymized.jpg
```

## Owner Selection / Face Re-ID

YOLO/tracking still owns detection and `track_id` assignment. This tool can now add the "do not anonymize this person" layer with InsightFace embeddings.

First, export source-frame face crops for the web UI:

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --owner-crops-dir local/owner_candidates
```

The command writes crop thumbnails plus `manifest.json`:

```json
{
  "source_frame": 0,
  "faces": [
    {
      "face_index": 1,
      "track_id": "3",
      "bbox": [120, 80, 220, 210],
      "crop_path": "frame_000000_face_001_track_3.jpg"
    }
  ]
}
```

After the user selects a thumbnail, create an owner embedding profile and run anonymization. `--owner-face-index` is zero-based and refers to the `faces[]` entry in that source frame.

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --owner-face-index 1 \
  --owner-profile local/owner_profile.json \
  --output local/anonymized.mp4 \
  --report-json local/anonymized_report.json
```

For later runs, load the saved profile without asking the user again:

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --owner-profile local/owner_profile.json \
  --output local/anonymized.mp4
```

Per frame, the tool crops each detected face, extracts an InsightFace embedding, compares it with the owner profile by cosine similarity, and dynamically adds matching `track_id`s to the keep set. This lets the owner be recovered even if the tracker changes IDs.

Default owner thresholds:

```text
similarity >= 0.55  owner vote
similarity <= 0.35  non-owner vote
between them        keep previous track state / temporal vote
```

Useful knobs:

```text
--owner-high-threshold 0.55
--owner-low-threshold 0.35
--owner-vote-window 10
--owner-min-votes 3
--owner-hold-frames 12
--face-recognition-model buffalo_l
--face-recognition-providers CPUExecutionProvider
```

The saved owner profile contains a biometric embedding. Keep it local and avoid putting it in logs or shared reports. `--report-json` omits embedding values by default; use `--include-owner-embedding-in-report` only when you deliberately need them.

## Reference Images

There are two levels of reference-image support.

Prompt-only reference extraction is light and safe, but weak for preserving a face:

```bash
python face-anonymizer/reference_prompt.py \
  --images refs/synthetic_01.jpg refs/synthetic_02.jpg \
  --output-json local/reference_prompt.json
```

Then load it during anonymization:

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --reference-prompt-json local/reference_prompt.json \
  --output local/anonymized.mp4
```

For stronger preservation, use reference-face conditioning. This feather-blends a synthetic/consented reference face into each target crop before SDXL harmonizes it:

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --reference-face-images refs/synthetic_01.jpg refs/synthetic_02.jpg \
  --strength 0.35 \
  --output local/anonymized.mp4
```

Use `--inpaint-scope face-crop` with this mode. Lower `--strength` values, around `0.25` to `0.45`, preserve the reference more; higher values anonymize more aggressively but drift away from the reference. Multiple reference images work best when they are the same synthetic identity under different lighting or angles. Mixing different identities weakens consistency.

For production routing, use `--reference-identity-images`. The tool first asks InsightFace whether the reference contains a valid human face:

```text
human face detected    -> InstantID route
no human face detected -> IP-Adapter Plus SDXL route
```

Human reference example:

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --reference-identity-images refs/synthetic_human_01.jpg refs/synthetic_human_02.jpg \
  --reference-route auto \
  --instantid-pipeline-dir external/InstantID \
  --instantid-adapter-path external/InstantID/checkpoints/ip-adapter.bin \
  --instantid-controlnet-model external/InstantID/checkpoints/ControlNetModel \
  --reference-face-model-root external/InstantID \
  --output local/anonymized.mp4
```

Character or animal reference example:

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --reference-identity-images refs/panda.png refs/panda_side.png \
  --reference-route auto \
  --reference-character-prompt "a realistic panda face, natural lighting" \
  --output local/anonymized.mp4
```

`auto` uses InsightFace/antelopev2 detection confidence and `--reference-human-min-ratio`. If you already know the reference type, force `--reference-route instantid` or `--reference-route ip-adapter`.

## Fire Anonymization Preset

For video, a flame/smoke replacement can hide diffusion flicker because real fire already changes shape frame to frame. The preset keeps the anonymization non-human by fixing the positive prompt to an opaque VFX fire mask and adding safety negatives for human likeness, face swap, deepfake, and recognizable-person cues.

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --output local/fire_anonymized.mp4 \
  --replacement-preset fire \
  --mask-mode bbox \
  --seed 1234 \
  --seed-strategy track \
  --strength 1.0 \
  --guidance-scale 6.0 \
  --num-inference-steps 12 \
  --report-json local/fire_report.json
```

With `--replacement-preset fire`, `--mask-mode auto` is forced to `bbox` by default. This keeps the flame mass locked to the YOLO/tracker box while letting the flame detail flicker naturally. The preset also widens and feathers the mask slightly so warm fire light can bleed into nearby hair and shoulders.

The preset also uses `--fire-prepaint white --fire-prepaint-region bbox` by default. Before diffusion, the original detection bbox is filled with a plain white patch, while the diffusion mask can be slightly larger, for example `--mask-expansion 1.1`. This reduces the model's tendency to reconstruct the original face without creating a large white halo around the feathered mask. You can disable it with `--fire-prepaint none`.

For desktop GPU experiments, weak structure guidance can help keep the effect inside the face area:

```bash
--controlnet canny --controlnet-scale 0.45
```

## Mask Preview

Run this first to verify that the selected tracks and mask shape are correct. It does not load SDXL.

```bash
python face-anonymizer/anonymize.py \
  --input input.jpg \
  --detections face-anonymizer/example_detections.json \
  --keep-track-ids 1 \
  --output local/face_preview.jpg \
  --save-mask local/face_mask.png \
  --mask-preview
```

`keep-track-ids` means "do not anonymize these identities." Every other detected face is masked.

## Image Anonymization

```bash
python face-anonymizer/anonymize.py \
  --input input.jpg \
  --detections detections.json \
  --keep-track-ids 1 \
  --output local/anonymized.jpg \
  --identity-bank face-anonymizer/identity_bank.example.json \
  --report-json local/anonymized_report.json \
  --seed 1234
```

By default, this runs `--inpaint-scope face-crop`. Each selected face is cropped, inpainted at a better SDXL working size, and pasted back. This usually looks better than full-frame inpainting for small faces.

Use one-pass full-frame inpainting when you want all selected faces to be generated together:

```bash
python face-anonymizer/anonymize.py \
  --input input.jpg \
  --detections detections.json \
  --keep-track-ids 1 \
  --output local/anonymized.jpg \
  --inpaint-scope full-frame \
  --seed 1234
```

## Short Video Demo

For a very short demo, cap the number of frames first.

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --keep-track-ids 1 \
  --output local/anonymized.mp4 \
  --identity-bank face-anonymizer/identity_bank.example.json \
  --max-frames 24 \
  --report-json local/anonymized_report.json \
  --seed 1234
```

The output video currently writes video frames only; audio is not copied.

For short video stability, `--seed-strategy track` is the default. With `--seed 1234`, the same `track_id` gets the same deterministic seed offset across frames.

## ControlNet

ControlNet is optional and should be tested on the Windows GPU machine.

```bash
python face-anonymizer/anonymize.py \
  --input input.mp4 \
  --detections detections.json \
  --keep-track-ids 1 \
  --output local/anonymized_canny.mp4 \
  --identity-bank face-anonymizer/identity_bank.example.json \
  --controlnet canny \
  --controlnet-scale 0.45 \
  --max-frames 24 \
  --seed 1234
```

Use `--controlnet depth` when face volume/pose preservation matters more than edge detail. Canny is lighter and a good first test.

## Synthetic Identity Bank

`--identity-bank` maps anonymized tracks to synthetic identity prompts and optional LoRAs. If a detection has `synthetic_id`, that identity is used. Otherwise, the tool assigns one deterministically from `track_id`.

```json
{
  "id": "synthetic_male_01",
  "prompt": "synthetic non-famous Korean man face, realistic candid portrait",
  "lora": "local/loras/synthetic_male_01",
  "lora_weight": 0.75,
  "seed_offset": 21001
}
```

LoRA training notes and Windows wrappers are in `face-anonymizer/LORA_TRAINING.md`.

## Quality Fallbacks

When SDXL fails or barely changes the masked area, the tool can fall back to blur or pixelation and record it.

```bash
python face-anonymizer/anonymize.py \
  --input input.jpg \
  --detections detections.json \
  --keep-track-ids 1 \
  --output local/anonymized.jpg \
  --fallback-mode blur \
  --report-json local/report.json \
  --seed 1234
```

The report includes `track_id`, assigned synthetic identity, mask area ratio, masked mean pixel delta, and fallback reason.

## Useful Options

```text
--anonymize-track-ids 2,3      Only anonymize these IDs.
--keep-track-ids 1             Keep these IDs untouched.
--mask-mode auto               Use polygon, landmark, then ellipse fallback.
--mask-mode segmentation       Require segmentation polygons unless fallback is enabled.
--mask-mode bbox               Use the detection bbox as a rectangular mask.
--mask-mode sam                Use the detection bbox as a SAM box prompt.
--mask-fallback ellipse        Use bbox ellipse when richer mask data is missing.
--inpaint-scope face-crop      Inpaint each selected face crop separately.
--inpaint-scope full-frame     Inpaint all selected masks in one full frame pass.
--crop-expansion 2.4           Context around each face crop.
--crop-min-size 512            Minimum face crop size before SDXL resizing.
--mask-expansion 1.35          Expand bbox before drawing the face ellipse.
--sam-local-files-only         Load SAM only from the local Hugging Face cache.
--mask-dilation 10             Grow the binary mask before feathering.
--mask-blur 16                 Feather mask edges for cleaner inpainting.
--max-side 1024                Resize long side for SDXL work size.
--scheduler dpmpp_2m_karras    Default photoreal scheduler.
--seed-strategy track          Stable per-track seed offsets.
--identity-bank PATH           Synthetic identities and optional LoRAs.
--controlnet canny|depth       Optional structural guidance.
--controlnet-scale 0.55        Strength of ControlNet conditioning.
--replacement-preset fire      Cover masked faces with flame/smoke VFX.
--fire-prepaint white          Blank the masked input area before fire generation.
--fire-prepaint-region bbox    Prepaint only the raw bbox, not the blurred mask.
--fire-force-bbox              Make fire preset auto masks use the bbox.
--fallback-mode blur|pixelate  Safety fallback for failed generations.
--report-json PATH             Save quality/fallback metrics.
--owner-crops-dir DIR          Export source-frame crop thumbnails for UI selection.
--owner-face-index 1           Build owner embedding from selected source-frame face.
--owner-profile PATH           Save/load owner embedding profile for Re-ID keep logic.
--owner-high-threshold 0.55    Similarity threshold for owner votes.
--owner-vote-window 10         Number of recent similarities used per track.
--reference-images IMG...      Convert reference image traits into prompt text.
--reference-face-images IMG... Blend reference face into target crop before inpainting.
--reference-identity-images IMG... Auto-route references to InstantID or IP-Adapter.
--reference-route auto          Choose instantid/ip-adapter from InsightFace detection.
--instantid-pipeline-dir DIR    Directory containing InstantID pipeline_stable_diffusion_xl_instantid.py.
--ip-adapter-weight-name NAME   SDXL IP-Adapter checkpoint for character/animal references.
--model-id MODEL               Override the SDXL inpainting checkpoint.
--lora PATH_OR_REPO            Optional synthetic identity LoRA.
--lora-weight 0.8              Blend LoRA strength.
```

## Next Step

For production-quality masks, have the YOLO/segmentation side emit `segmentation` polygons or `landmarks`. The code will consume them directly and only falls back to bbox ellipses when richer mask data is missing.

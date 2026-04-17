# Test Report: Coffee Shop Dynamic Perception Demo

**Date**: 2026-04-12
**Spec**: (inline task specification, no pipeline_spec.md present)
**Implementation**: demo/coffee-shop-demo/

## Summary

| Result | Status |
|--------|--------|
| Runs to completion | PASS |
| Outputs produced | PASS |
| Schema valid | PASS |
| Content quality | PASS |
| Error handling | PASS |
| **Overall** | **PASS** |

All 7 unit tests passed on round 1. No fix rounds were required. Integration test (30-frame synthetic video) completed with exit code 0.

---

## Test A — Import and syntax checks

All imports succeeded using the venv Python interpreter.

```
python -c "import pipeline; print('pipeline OK')"
-> pipeline OK

python -c "from phases import common, phase0_baseline, phase1_dataset, phase2_finetune, phase3_evaluate; print('phases OK')"
-> phases OK

python run_plan.py --help
-> Subcommands listed correctly: status, phase0, phase1, phase2, download-model, phase3

python run_plan.py status
-> Checklist printed; phase0 marked complete, phases 1-3 pending
```

Note: `python` is not on PATH in this environment — the venv at `venv/bin/python` must be used explicitly, or a `python` symlink added for convenience.

---

## Test B — pipeline.py unit tests (7/7 passed)

Script: `_test_pipeline.py` (synthetic numpy frames, no video file needed)

| # | Test | Status |
|---|------|--------|
| 1 | Instantiation — `DynamicPerceptionPipeline(DEFAULT_CONFIG)` succeeds | PASS |
| 2 | `process_frame` frame 0 — returns `FrameResult` with all required fields including `context_objects` and `semantic_context` in `scene_state` | PASS |
| 3 | Frames 0-10 — flow fires at frame 5, carry-forward on 6-9, recomputes at 10 | PASS |
| 4 | `ContextObject` tracking — field exists on `FrameResult`, is a list, dataclass fields present | PASS |
| 5 | `SemanticInferenceEngine.infer()` — returns dict with keys: `occupied_tables`, `empty_chairs`, `inferred_occupancy_regions`, `furniture`, `context_object_count`; Rule 1 (cup near table -> occupied) verified | PASS |
| 6 | `predict_trajectory` — accepts both tuple list and dict list; short history returns `[]` | PASS |
| 7 | Bad frame shape — 2D grayscale and 4-channel frames both raise `ValueError` | PASS |

Pipeline uses RAFT-Small for optical flow. On CPU, 30 frames process at ~10.8 fps.

---

## Test C — phase1_dataset imports and URL reachability

```
UCY: HTTP 200 OK    (https://zenodo.org/api/records/10629757/files/Kornos_labeling.zip/content)
Mall: HTTP 200 OK   (https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip)
```

Both dataset URLs are reachable via HEAD request.

---

## Test D — run_plan.py CLI surface

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| `phase0 --help` — `--no-roboflow` absent | Not present | Not present | PASS |
| `phase1 --help` — `--no-roboflow` present | Present | Present | PASS |
| `download-model --help` — command exists | Exists | Exists with `--roboflow-key`, `--roboflow-project`, `--version` | PASS |
| `phase2 --help` — exists | Exists | Exists with `--epochs`, `--imgsz`, `--device`, `--batch` | PASS |
| `phase3 --help` — exists | Exists | Exists with `--video`, `--device`, `--with-gt` | PASS |

---

## Test E — Integration: 30-frame synthetic video through run_video.py

**Input**: `/tmp/cs_test.mp4` — 640x480, 30 fps, 30 frames (random noise + moving green rectangle)
**Command**: `python run_video.py --input /tmp/cs_test.mp4 --output /tmp/cs_out.mp4 --log /tmp/cs_log.jsonl --device cpu`

- Exit code: 0
- Output video `/tmp/cs_out.mp4`: produced, 3.5 MB
- JSONL log `/tmp/cs_log.jsonl`: 30 lines (one per frame)
- All 30 JSONL lines contain `context_objects` key: YES
- All 30 JSONL lines contain `semantic_context` key: YES
- `scene_state` keys per frame: `frame_idx`, `timestamp_s`, `objects`, `unclassified_motion_regions`, `flow_computed_this_frame`, `dynamic_pixel_count`, `nearest_person_px`, `context_objects`, `semantic_context`
- `semantic_context` keys: `occupied_tables`, `empty_chairs`, `inferred_occupancy_regions`, `furniture`, `context_object_count`
- Processing throughput: ~10.8 fps on CPU (2.8 s for 30 frames)

---

## Step-by-step log (integration run stdout)

```
[run_video] Opening: /tmp/cs_test.mp4
[run_video] Video: 640x480 @ 30.0 fps, 30 frames
[run_video] Scene log: /tmp/cs_log.jsonl
[run_video] Initialising pipeline...
[pipeline] Using device: cpu
Processing: 100%|██████████| 30/30 [00:02<00:00, 10.82frame/s]

[run_video] Done. Processed 30 frames in 2.8s (10.8 fps average).
[run_video] Output video: /private/tmp/cs_out.mp4
[run_video] Scene log: /private/tmp/cs_log.jsonl
```

---

## Output validation

All expected outputs are produced and schema-compliant:

1. **`FrameResult`** — `annotated_frame` (BGR ndarray), `scene_state` (dict), `dynamic_mask` (uint8 ndarray), `context_objects` (list of `ContextObject`)
2. **`scene_state` dict** — all required keys present including `context_objects` and `semantic_context`
3. **`semantic_context` dict** — all 5 required keys present: `occupied_tables`, `empty_chairs`, `inferred_occupancy_regions`, `furniture`, `context_object_count`
4. **JSONL log** — 30 lines, each a valid JSON object with all expected keys
5. **Output video** — produced, non-zero size (3.5 MB)

---

## Failures and errors

None. All tests passed in round 1 with no fixes required.

---

## Spec ambiguities observed

The following `SPEC_AMBIGUITY` comment is present in `pipeline.py`:

> `_ANCHOR_CLASSES = [56, 41]  # chair, cup — reserved for future use`
> SPEC_AMBIGUITY: spec says "use chairs(56)/cups(41) as static anchors" but does not define any downstream use for anchor detections. Defined for completeness but no separate anchor pass is run.

This did not cause any test failures. Context objects including chairs and cups are handled through the full context-class tracking path.

---

## Recommendations

1. **PATH / venv ergonomics**: The system `python` is not on PATH (only `python3` is). The venv must be activated or `venv/bin/python` used directly. The README should document this explicitly, or a `Makefile` target / `run.sh` wrapper should activate the venv before running any command.

2. **Dead colour constants**: The `_draw_annotations` method defines several colour constants (`BLUE`, `ORANGE`, `SOLID_BLUE`, `TRUE_BLUE`, `COLOR_OCC_SIGNAL_BLUE`) that are never used. These are dead code and should be removed to avoid confusion for future contributors.

3. **`--with-gt` is TODO**: `phase3 --help` exposes `--with-gt` flagged as `[TODO]`. If ground-truth evaluation is part of the spec, this should either be implemented or removed from the public CLI surface.

4. **`_ANCHOR_CLASSES` is dead code**: `_ANCHOR_CLASSES = [56, 41]` is defined but never referenced. Either document its future role or remove it to keep the codebase clean.

5. **`_SemanticInferenceEngine` naming**: The class is exported as `SemanticInferenceEngine` (no leading underscore), which is correct for a public/testable symbol. The pipeline attribute is named `_semantic_engine` (private). This is consistent; just ensure external test authors know to import `SemanticInferenceEngine` directly rather than accessing `pipeline._semantic_engine`.

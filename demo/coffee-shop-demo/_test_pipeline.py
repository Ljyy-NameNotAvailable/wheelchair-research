"""
_test_pipeline.py — Unit tests for pipeline.py (no video file needed).
Runs with synthetic numpy frames only.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
import numpy as np

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    DEFAULT_CONFIG,
    DynamicPerceptionPipeline,
    FrameResult,
    ContextObject,
    SemanticInferenceEngine,
    TrackedObject,
)

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []  # (test_name, status, detail)


def check(name: str, fn) -> bool:
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  [PASS] {name}")
        return True
    except Exception as exc:
        tb = traceback.format_exc()
        results.append((name, FAIL, tb))
        print(f"  [FAIL] {name}\n         {exc}")
        return False


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Synthetic random BGR frame."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Instantiate pipeline once (shared across tests)
# ---------------------------------------------------------------------------
print("\n=== Setting up pipeline (this may take a moment for RAFT) ===")
config = {**DEFAULT_CONFIG, "device": "cpu"}
try:
    pipeline = DynamicPerceptionPipeline(config)
    print("Pipeline instantiated OK")
except Exception as e:
    print(f"FATAL: Could not instantiate pipeline: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Running tests ===\n")


# ---------------------------------------------------------------------------
# Test 1: Instantiation
# ---------------------------------------------------------------------------
def test_instantiation():
    cfg = {**DEFAULT_CONFIG, "device": "cpu"}
    p = DynamicPerceptionPipeline(cfg)
    assert p is not None
    assert hasattr(p, "cfg")
    assert hasattr(p, "yolo")
    assert hasattr(p, "_raft")
    assert hasattr(p, "_semantic_engine")

check("1. Instantiation", test_instantiation)


# ---------------------------------------------------------------------------
# Test 2: process_frame frame 0 returns FrameResult with correct fields
# ---------------------------------------------------------------------------
def test_frame0():
    p = DynamicPerceptionPipeline({**DEFAULT_CONFIG, "device": "cpu"})
    frame = _make_frame()
    result = p.process_frame(frame, frame_idx=0)

    assert isinstance(result, FrameResult), f"Expected FrameResult, got {type(result)}"
    assert isinstance(result.annotated_frame, np.ndarray), "annotated_frame must be ndarray"
    assert result.annotated_frame.shape == frame.shape, \
        f"annotated_frame shape {result.annotated_frame.shape} != {frame.shape}"
    assert isinstance(result.scene_state, dict), "scene_state must be dict"
    assert isinstance(result.dynamic_mask, np.ndarray), "dynamic_mask must be ndarray"
    assert isinstance(result.context_objects, list), "context_objects must be list"

    # Check scene_state contains context_objects and semantic_context
    assert "context_objects" in result.scene_state, \
        "scene_state missing 'context_objects' key"
    assert "semantic_context" in result.scene_state, \
        "scene_state missing 'semantic_context' key"
    assert isinstance(result.scene_state["context_objects"], list)
    assert isinstance(result.scene_state["semantic_context"], dict)

check("2. process_frame frame 0 — FrameResult fields + context_objects + semantic_context", test_frame0)


# ---------------------------------------------------------------------------
# Test 3: process_frame frames 0-10, flow fires at frame 5, carry-forward on 6-9
# ---------------------------------------------------------------------------
def test_frames_0_to_10():
    p = DynamicPerceptionPipeline({**DEFAULT_CONFIG, "device": "cpu", "flow_interval": 5})
    results_list = []
    for i in range(11):
        frame = _make_frame()
        r = p.process_frame(frame, frame_idx=i)
        results_list.append(r)

    # Frame 0: no prev_frame, flow not computed
    assert results_list[0].scene_state["flow_computed_this_frame"] is False, \
        "Frame 0 should not compute flow (no prev frame)"

    # Frame 5: first flow computation (flow_interval=5, frame_idx=5)
    assert results_list[5].scene_state["flow_computed_this_frame"] is True, \
        "Frame 5 should compute flow"

    # Frame 6-9: flow not recomputed, carried forward (dynamic_mask still exists)
    for i in range(6, 10):
        assert results_list[i].scene_state["flow_computed_this_frame"] is False, \
            f"Frame {i} should NOT compute flow (carry-forward)"

    # Frame 10: 10 % 5 == 0, should compute flow again
    assert results_list[10].scene_state["flow_computed_this_frame"] is True, \
        "Frame 10 should compute flow again (10 % 5 == 0)"

check("3. Frames 0-10 — flow fires at 5, carry-forward 6-9, recomputes at 10", test_frames_0_to_10)


# ---------------------------------------------------------------------------
# Test 4: ContextObject tracking field exists and is a list
# ---------------------------------------------------------------------------
def test_context_objects_field():
    p = DynamicPerceptionPipeline({**DEFAULT_CONFIG, "device": "cpu"})
    for i in range(6):
        frame = _make_frame()
        result = p.process_frame(frame, frame_idx=i)

    # context_objects field must exist and be a list
    assert isinstance(result.context_objects, list), \
        f"context_objects must be list, got {type(result.context_objects)}"

    # If any context objects are detected, verify fields
    for co in result.context_objects:
        assert isinstance(co, ContextObject), f"Expected ContextObject, got {type(co)}"
        assert hasattr(co, "track_id")
        assert hasattr(co, "class_name")
        assert hasattr(co, "class_id")
        assert hasattr(co, "bbox_px")
        assert hasattr(co, "centroid_px")
        assert hasattr(co, "frames_tracked")
        assert hasattr(co, "is_furniture")
        assert hasattr(co, "is_occupancy_signal")

check("4. ContextObject tracking — field exists, is list, correct structure", test_context_objects_field)


# ---------------------------------------------------------------------------
# Test 5: SemanticInferenceEngine.infer() returns dict with required keys
# ---------------------------------------------------------------------------
def test_semantic_inference_engine():
    engine = SemanticInferenceEngine(proximity_px=120, min_frames=5)

    # Build mock data with a table and a cup nearby
    mock_table = ContextObject(
        track_id=1, class_name="dining table", class_id=60,
        bbox_px=[100, 100, 300, 300], centroid_px=[200, 200],
        confidence=0.9, frames_tracked=10,
        is_furniture=True, is_occupancy_signal=False,
    )
    mock_cup = ContextObject(
        track_id=2, class_name="cup", class_id=41,
        bbox_px=[190, 190, 230, 230], centroid_px=[210, 210],
        confidence=0.85, frames_tracked=7,
        is_furniture=False, is_occupancy_signal=True,
    )
    mock_chair = ContextObject(
        track_id=3, class_name="chair", class_id=56,
        bbox_px=[400, 400, 500, 500], centroid_px=[450, 450],
        confidence=0.8, frames_tracked=3,
        is_furniture=True, is_occupancy_signal=False,
    )

    mock_person = TrackedObject(
        track_id=10, class_name="person",
        bbox_px=[50, 50, 150, 250], centroid_px=[100, 150],
        confidence=0.92,
    )

    result = engine.infer(
        context_objects=[mock_table, mock_cup, mock_chair],
        tracked_persons=[mock_person],
        frame_shape=(480, 640),
    )

    assert isinstance(result, dict), f"infer() must return dict, got {type(result)}"

    required_keys = [
        "occupied_tables",
        "empty_chairs",
        "inferred_occupancy_regions",
        "furniture",
        "context_object_count",
    ]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in semantic inference result"

    # The cup is near the table and stable (frames_tracked=7 >= min_frames=5)
    # so occupied_tables should be 1
    assert result["occupied_tables"] == 1, \
        f"Expected occupied_tables=1 (cup near table), got {result['occupied_tables']}"

    # context_object_count should match input
    assert result["context_object_count"] == 3, \
        f"Expected context_object_count=3, got {result['context_object_count']}"

    # furniture should have entries for table and chair
    assert len(result["furniture"]) == 2, \
        f"Expected 2 furniture entries, got {len(result['furniture'])}"

check("5. SemanticInferenceEngine.infer() — keys + occupied_tables logic", test_semantic_inference_engine)


# ---------------------------------------------------------------------------
# Test 6: predict_trajectory accepts tuple list and dict list
# ---------------------------------------------------------------------------
def test_predict_trajectory():
    p = DynamicPerceptionPipeline({**DEFAULT_CONFIG, "device": "cpu"})

    # Tuple list input
    history_tuples = [(100, 100), (105, 103), (110, 106)]
    result_tuples = p.predict_trajectory(history_tuples, steps=3)
    assert isinstance(result_tuples, list), "Must return list"
    assert len(result_tuples) == 3, f"Expected 3 steps, got {len(result_tuples)}"
    assert all(len(pt) == 2 for pt in result_tuples), "Each prediction must be (x, y)"

    # Dict list input
    history_dicts = [
        {"centroid_px": [100, 100]},
        {"centroid_px": [105, 103]},
        {"centroid_px": [110, 106]},
    ]
    result_dicts = p.predict_trajectory(history_dicts, steps=3)
    assert isinstance(result_dicts, list)
    assert len(result_dicts) == 3

    # Both should produce same results
    assert result_tuples == result_dicts, \
        f"Tuple and dict inputs should give same result: {result_tuples} vs {result_dicts}"

    # Short history — expect empty
    result_short = p.predict_trajectory([(100, 100)], steps=3)
    assert result_short == [], f"Single-point history should return empty list, got {result_short}"

check("6. predict_trajectory — tuple list and dict list inputs", test_predict_trajectory)


# ---------------------------------------------------------------------------
# Test 7: Bad frame shape raises ValueError
# ---------------------------------------------------------------------------
def test_bad_frame_shape():
    p = DynamicPerceptionPipeline({**DEFAULT_CONFIG, "device": "cpu"})

    # 2D grayscale — should raise ValueError
    bad_frame = np.zeros((480, 640), dtype=np.uint8)
    raised = False
    try:
        p.process_frame(bad_frame, frame_idx=0)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for 2D (grayscale) frame"

    # 4-channel RGBA — should also raise ValueError
    bad_frame_4ch = np.zeros((480, 640, 4), dtype=np.uint8)
    raised4 = False
    try:
        p.process_frame(bad_frame_4ch, frame_idx=0)
    except ValueError:
        raised4 = True
    assert raised4, "Expected ValueError for 4-channel frame"

check("7. Bad frame shape raises ValueError", test_bad_frame_shape)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== SUMMARY ===")
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
total = len(results)
print(f"  {passed}/{total} tests passed, {failed} failed")
for name, status, detail in results:
    mark = "PASS" if status == PASS else "FAIL"
    print(f"  [{mark}] {name}")
    if detail:
        # Print first line of traceback
        lines = [l for l in detail.strip().splitlines() if l.strip()]
        if lines:
            print(f"         -> {lines[-1]}")

sys.exit(0 if failed == 0 else 1)

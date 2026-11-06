"""Analyze one video from start to finish for the dashboard.

This is the inference version of the training pipeline: four pose passes, blending, quality
control, LMA features, then the model on sliding windows. It all runs in one process, on the
modules copied from training-pipeline/ (see scripts/sync_dashboard.py).
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


APP_DIR = Path(__file__).resolve().parent
ROOT_CANDIDATES = (APP_DIR, APP_DIR.parent)
PROJECT_ROOT = next((p for p in ROOT_CANDIDATES if (p / "src").exists()), APP_DIR.parent)
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = next((p / "training" for p in ROOT_CANDIDATES if (p / "training").exists()), PROJECT_ROOT / "training")

# Prepend, so the project's own packages (utils, config, ...) win over same-named installed ones.
for path in (APP_DIR, PROJECT_ROOT, SRC_DIR, TRAINING_DIR):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import PoseEstimationConfig  # type: ignore  # noqa: E402
from features import compute_lma_features  # type: ignore  # noqa: E402
from processing.quality import (  # type: ignore  # noqa: E402
    QualityControlConfig,
    apply_quality_control,
)
from training.gesture.gesture_dataset import DanceEmotionDataset  # type: ignore  # noqa: E402
from training.gesture.model import DualStreamEmotionTransformer  # type: ignore  # noqa: E402
from utils.io import serialise_pose_run  # type: ignore  # noqa: E402
from utils.pose_utils import estimate_pose_landmarks  # type: ignore  # noqa: E402
from utils.video_utils import iter_video_frames, read_video_fps  # type: ignore  # noqa: E402

# Analyzed frames per second, for checkpoints that don't record their own: the models of this
# project were trained on 50 fps videos, sampled one frame in four.
DEFAULT_SAMPLE_RATE = 12.5

# Hips and legs (landmarks 23 to 32). Every training clip has at least 52 % of them visible,
# so a video under this share is unlike anything the model has seen.
LOWER_BODY_LANDMARKS = range(23, 33)
MIN_LOWER_BODY_VISIBLE = 0.5
# Under this share of sampled frames with a pose, the person is missing from much of the video.
MIN_DETECTED_SHARE = 0.5

UNREADABLE_VIDEO = "This file could not be read as a video. Check that it plays, or convert it to MP4."
NO_PERSON = "No person was detected. Use a video where one person's whole body is clearly visible."


@dataclass
class ModelInfo:
    """A loaded model, with the label maps and window settings it was trained with."""
    path: Path
    model: DualStreamEmotionTransformer
    idx_to_emotion: Dict[int, str]
    emotion_to_idx: Dict[str, int]
    max_len: int
    window_stride: int
    cfg: dict


@dataclass
class WindowPrediction:
    """Emotion probabilities of one window. cumulative_probs averages all windows so far."""
    start_idx: int
    start_time_s: float
    end_time_s: float
    probs: Dict[str, float]
    cumulative_probs: Dict[str, float]


@dataclass
class InferenceResult:
    """Everything the dashboard needs to show one analysis."""
    video_path: Path
    pose_json: Path
    features_json: Path
    frame_stride: float
    fps: float
    window_len: int
    window_stride: int
    windows: List[WindowPrediction]
    avg_probs: Dict[str, float]
    notices: List[str] = field(default_factory=list)


def list_available_checkpoints(extra_dirs: Optional[Sequence[Path]] = None) -> List[Path]:
    """Return checkpoint.pt files under outputs/models*, plus any extras."""

    roots = [
        PROJECT_ROOT / "outputs" / "models",
        PROJECT_ROOT / "outputs" / "models_pull",
    ]
    if extra_dirs:
        roots.extend(extra_dirs)

    checkpoints: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for folder in sorted(root.iterdir()):
            candidate = folder / "checkpoint.pt"
            if candidate.exists():
                checkpoints.append(candidate)
    return checkpoints


def load_checkpoint(ckpt_path: Path, device: torch.device) -> ModelInfo:
    """Load a dual-stream checkpoint and return the model plus metadata."""

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with torch.serialization.safe_globals([pathlib.PosixPath]):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg = ckpt["cfg"]
    emo_to_idx = ckpt["emotion_to_idx"]
    idx_to_emo = ckpt["idx_to_emotion"]

    model = DualStreamEmotionTransformer(
        pose_dim=33 * 3,
        laban_dim=6,
        num_emotions=len(emo_to_idx),
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        max_len=cfg["model"]["max_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_len = int(cfg["model"]["max_len"])
    args = ckpt.get("args", {}) or {}
    window_stride = (
        int(args.get("window_stride"))
        if args.get("window_stride") not in (None, "", 0)
        else None
    )
    if window_stride is None:
        window_stride = int(cfg.get("training", {}).get("stride", max_len))

    return ModelInfo(
        path=ckpt_path,
        model=model,
        idx_to_emotion=idx_to_emo,
        emotion_to_idx=emo_to_idx,
        max_len=max_len,
        window_stride=window_stride or max_len,
        cfg=cfg,
    )


def get_model_config(ckpt_path: Path) -> dict:
    """Lightweight loader to get just the model configuration/metadata."""
    if not ckpt_path.exists():
        return {}

    with torch.serialization.safe_globals([pathlib.PosixPath]):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    cfg = ckpt.get("cfg", {})
    args = ckpt.get("args", {}) or {}

    max_len = int(cfg.get("model", {}).get("max_len", 0))
    window_stride = int(args.get("window_stride") or cfg.get("training", {}).get("stride", max_len))
    sample_rate = float(ckpt.get("sample_rate") or DEFAULT_SAMPLE_RATE)

    return {
        "max_len": max_len,
        "window_stride": window_stride,
        "sample_rate": sample_rate,
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Softmax of one vector. Subtracting the maximum first avoids overflow."""
    m = logits.max()
    exps = np.exp(logits - m)
    return exps / exps.sum()


def _window_starts(total: int, window_len: int, stride: int) -> List[int]:
    """Return the window start indices. The last window is aligned to the end of the clip."""
    if total <= 0:
        return []
    starts = list(range(0, max(total - window_len, 0) + 1, stride))
    if starts and starts[-1] + window_len < total:
        starts.append(total - window_len)
    if not starts:
        starts = [0]
    return [int(s) for s in starts]


import subprocess
from dataclasses import replace
from utils.types import FrameDiagnostics, FramePose

def convert_video_to_mp4(input_path: Path) -> Path:
    """
    Convert video to browser-friendly MP4 (H.264/AAC) using ffmpeg.
    Returns the path to the converted video.
    """
    output_path = input_path.with_suffix(".mp4")

    if output_path.resolve() == input_path.resolve():
        output_path = input_path.with_name(input_path.stem + "_converted.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-ac", "2",
        "-strict", "-2",
        # Browsers need 4:2:0 color, and that format requires an even width and height.
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg was not found. Install it, then restart the dashboard.")
    except subprocess.CalledProcessError:
        raise RuntimeError(UNREADABLE_VIDEO)

    return output_path


def blend_pose_results(
    runs: Dict[str, tuple[List[FramePose], List[FrameDiagnostics]]],
    metric: str = "world_visible_ratio",
    min_delta: float = 0.0
) -> tuple[List[FramePose], List[FrameDiagnostics]]:
    """
    Merge multiple pose runs by selecting the best frame per index.
    Logic adapted from src/tools/combine_pose_runs.py.
    """

    def compute_metric(diag: FrameDiagnostics, metric: str) -> float:
        if metric == "world_visible_ratio":
            return (diag.world_landmarks_visible / diag.world_landmarks_total) if diag.world_landmarks_total else 0.0

        return 0.0

    all_indices = set()
    run_data = {}

    for label, (frames, diags) in runs.items():
        pose_map = {f.frame_index: f for f in frames}
        diag_map = {d.frame_index: d for d in diags}
        all_indices.update(pose_map.keys())
        run_data[label] = (pose_map, diag_map)

    sorted_indices = sorted(all_indices)
    merged_frames = []
    merged_diagnostics = []

    current_label = None

    for idx in sorted_indices:
        candidates = []
        for label, (pose_map, diag_map) in run_data.items():
            diag = diag_map.get(idx)
            pose = pose_map.get(idx)
            if diag and pose:
                score = compute_metric(diag, metric)
                candidates.append((score, label, diag, pose))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)

        best = candidates[0]

        chosen = best
        # Same rule as combine_pose_runs.py: with min_delta 0.0, the best run always wins.
        if current_label:
            current_cand = next((c for c in candidates if c[1] == current_label), None)
            if current_cand:
                if best[1] == current_label or (best[0] - current_cand[0]) >= min_delta:
                    chosen = best
                else:
                    chosen = current_cand

        score, label, diag, pose = chosen

        merged_frames.append(pose)

        new_diag = replace(diag, fallback_reason=f"{diag.fallback_reason or 'ok'}|selected={label}")
        merged_diagnostics.append(new_diag)

        current_label = label

    return merged_frames, merged_diagnostics


def run_multi_pass_inference(
    video_path: Path,
    *,
    ckpt: Path,
    output_dir: Path,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> InferenceResult:
    """
    Mimics run_full_pipeline.py:
    1. Runs 4 passes: raw, segmentation, denoise, illumination
    2. Blends results
    3. Runs QC & Features
    4. Runs Model Inference

    `sample_rate` is in analyzed frames per second. Sampling by time, not by a fixed stride, gives
    the model the time scale it was trained on, whatever the frame rate of the video.

    `progress` receives a short message at each stage, for the dashboard's progress view.
    """
    report = progress or (lambda message: None)

    try:
        fps = read_video_fps(video_path)
    except FileNotFoundError:
        raise RuntimeError(UNREADABLE_VIDEO)
    frame_stride = max(fps / sample_rate, 1.0)

    # capture_image keeps the 2D landmarks too, like the training pipeline (--include-2d); the
    # dashboard draws them as the skeleton overlay. Predictions only use the 3D landmarks.
    pass_specs = [
        ("raw", PoseEstimationConfig(
            max_frames=max_frames, capture_image=True,
            preprocess_mode="segmentation",
            enable_person_crop=False,
            background_blur=False
        )),
        ("segmentation", PoseEstimationConfig(
            max_frames=max_frames, capture_image=True,
            preprocess_mode="segmentation",
            min_crop_area=0.02, segmentation_confidence=0.05, crop_padding=0.2
        )),
        ("denoise", PoseEstimationConfig(
            max_frames=max_frames, capture_image=True,
            preprocess_mode="denoise_sharpen",
            min_crop_area=0.02, segmentation_confidence=0.05, crop_padding=0.2
        )),
        ("illumination", PoseEstimationConfig(
            max_frames=max_frames, capture_image=True,
            preprocess_mode="illumination",
            min_crop_area=0.02, segmentation_confidence=0.05, crop_padding=0.2
        ))
    ]

    runs_results = {}

    for number, (label, cfg) in enumerate(pass_specs, start=1):
        report(f"Detecting poses · pass {number} of {len(pass_specs)}")
        cfg.save_frames = False
        try:
            # Each pass reads the video again, so a long video never has to fit in memory.
            samples = iter_video_frames(video_path, frame_stride, max_samples=max_frames)
            frames, diags = estimate_pose_landmarks(samples, cfg, frame_paths=None)
            runs_results[label] = (frames, diags)
        except Exception as e:
            print(f"Warning: Pass {label} failed: {e}")

    # The metrics cover every sampled frame, even those without a pose.
    sampled_frames = max((len(diags) for _, diags in runs_results.values()), default=0)
    if not sampled_frames:
        raise RuntimeError(UNREADABLE_VIDEO)

    report("Analyzing movement")
    merged_frames, merged_diags = blend_pose_results(runs_results)
    if not merged_frames:
        raise RuntimeError(NO_PERSON)

    # Same settings as training-pipeline/src/run_full_pipeline.py, which produced the training data.
    qc_config = QualityControlConfig(
        min_visibility_world=0.0,
        min_visibility_image=0.0,
        min_valid_landmarks_ratio=0.6,
        max_joint_speed=0.38,
        gap_fill_frames=5,
    )

    qc_frames, qc_report = apply_quality_control(merged_frames, qc_config)

    # Training leaves out landmarks under 0.5 visibility here (--min-visibility-world) and the
    # dashboard keeps them all. Measured on the 31 test clips, no window changes emotion.
    features = compute_lma_features(qc_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    pose_json = output_dir / "poses.json"
    features_json = output_dir / "features.json"

    payload = serialise_pose_run(
        video_path=video_path,
        stride=frame_stride,
        fps=fps,
        frames=qc_frames,
        frame_metrics=merged_diags,
        visibility_threshold=0.0,
    )
    payload["sampled_frames"] = sampled_frames
    with pose_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    feature_payload = {
        "video_path": str(video_path),
        "fps": fps,
        "frame_stride": frame_stride,
        "num_frames": len(features),
        "features": [feat.as_dict() for feat in features],
    }
    with features_json.open("w", encoding="utf-8") as f:
        json.dump(feature_payload, f, indent=2)

    report("Predicting emotions")
    return run_inference_from_features(
        video_path=video_path,
        pose_json=pose_json,
        features_json=features_json,
        ckpt=ckpt,
        output_dir=output_dir,
        frame_stride=frame_stride,
        fps=fps,
        device=device
    )


def quality_notices(pose_data: dict, window_len: int) -> List[str]:
    """Explain why the result of this video may be less reliable, if there is a reason to."""
    notices = []
    frames = pose_data.get("frames", [])
    sampled = pose_data.get("sampled_frames") or len(frames)
    fps = float(pose_data.get("fps") or 0)
    if frames and len(frames) < window_len and fps:
        seconds = window_len * float(pose_data.get("frame_stride") or 1) / fps
        notices.append(f"the clip is shorter than one analysis window ({seconds:.0f} s).")
    if frames and len(frames) / sampled < MIN_DETECTED_SHARE:
        notices.append(f"the person was found in only {len(frames) / sampled:.0%} of the video.")
    lower = [
        lm["visibility"] >= 0.5
        for frame in frames
        for lm in frame.get("landmarks_world") or []
        if lm["index"] in LOWER_BODY_LANDMARKS
    ]
    if lower and sum(lower) / len(lower) < MIN_LOWER_BODY_VISIBLE:
        notices.append("the hips and legs are mostly out of view.")
    return notices


def run_inference_from_features(
    video_path: Path,
    pose_json: Path,
    features_json: Path,
    ckpt: Path,
    output_dir: Path,
    frame_stride: float = 4.0,
    fps: float = 30.0,
    device: Optional[str] = None
) -> InferenceResult:
    """Run ONLY the model inference part, reusing existing features."""

    with pose_json.open("r") as f:
        pose_data = json.load(f)
    frames_count = len(pose_data.get("frames", []))
    # The pose file records how the video was sampled; prefer it for the timeline.
    fps = float(pose_data.get("fps") or fps)
    frame_stride = float(pose_data.get("frame_stride") or frame_stride)

    with features_json.open("r") as f:
        feat_data = json.load(f)
    features_count = feat_data.get("num_frames", 0)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_info = load_checkpoint(ckpt, torch_device)

    total_frames = min(frames_count, features_count) if features_count > 0 else frames_count

    starts = _window_starts(total_frames, model_info.max_len, model_info.window_stride)

    if not starts:
        raise RuntimeError("Too few frames with a visible person to analyze this video.")

    rows = []
    # The dataset requires a label, but it plays no role at inference.
    dummy_emotion = next(iter(model_info.emotion_to_idx))
    for s in starts:
        rows.append({
            "video_path": str(video_path),
            "pose_json": str(pose_json),
            "features_json": str(features_json),
            "emotion": dummy_emotion,
            "start_idx": int(s),
        })

    df = pd.DataFrame(rows)
    ds = DanceEmotionDataset(
        dataframe=df,
        emotion_to_idx=model_info.emotion_to_idx,
        max_len=model_info.max_len,
        visibility_threshold=0.0,
        replace_invalid_with_zeros=False,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    windows: List[WindowPrediction] = []
    # Two running sums: the probabilities give the "So far" reading, which shows the mix of emotions,
    # and the logits give the whole-video verdict.
    running_probs = np.zeros(len(model_info.idx_to_emotion), dtype=np.float64)
    running_logits = np.zeros(len(model_info.idx_to_emotion), dtype=np.float64)

    with torch.inference_mode():
        for pose_seq, laban_seq, mask, _labels, infos in loader:
            pose_seq = pose_seq.to(torch_device)
            laban_seq = laban_seq.to(torch_device)
            mask = mask.to(torch_device)

            logits = model_info.model(pose_seq, laban_seq, mask=mask)
            probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
            logits_batch = logits.cpu().numpy()

            keys = list(infos.keys())
            batch_size = len(next(iter(infos.values())))
            batch_infos = [{k: infos[k][i] for k in keys} for i in range(batch_size)]

            for i, info in enumerate(batch_infos):
                start_idx = int(info.get("start_idx", 0))
                prob_vec = probs_batch[i]
                running_probs += prob_vec
                running_logits += logits_batch[i]
                count = len(windows) + 1
                cum = running_probs / float(count)

                # Windows count sampled frames; the stride converts them back to video frames.
                start_time = (start_idx * frame_stride) / float(fps or 1.0)
                end_time = ((start_idx + model_info.max_len) * frame_stride) / float(fps or 1.0)

                probs = {model_info.idx_to_emotion[j]: float(prob_vec[j]) for j in range(len(prob_vec))}
                cum_probs = {model_info.idx_to_emotion[j]: float(cum[j]) for j in range(len(cum))}

                windows.append(WindowPrediction(
                    start_idx=start_idx,
                    start_time_s=start_time,
                    end_time_s=end_time,
                    probs=probs,
                    cumulative_probs=cum_probs
                ))

    # Whole-video verdict: the softmax of the mean window logits, the method behind the published
    # scores (see eval_gesture.py). Confident windows weigh more here than in "So far", so these
    # values don't describe the mix of emotions: the dashboard only shows the top emotion.
    whole_video = _softmax_np(running_logits / max(len(windows), 1))
    avg_probs = {model_info.idx_to_emotion[j]: float(whole_video[j]) for j in range(len(whole_video))}

    return InferenceResult(
        video_path=video_path,
        pose_json=pose_json,
        features_json=features_json,
        frame_stride=frame_stride,
        fps=fps,
        window_len=model_info.max_len,
        window_stride=model_info.window_stride,
        windows=windows,
        avg_probs=avg_probs,
        notices=quality_notices(pose_data, model_info.max_len),
    )

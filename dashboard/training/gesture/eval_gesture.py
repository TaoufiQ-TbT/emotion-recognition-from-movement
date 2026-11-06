"""Evaluate a checkpoint per clip and save every prediction to a JSON report.

Clips are cut into windows as in training. The window logits of a clip are averaged, and the
highest average gives its prediction. This is the method behind the scores in the main README.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from training.gesture.gesture_dataset import DanceEmotionDataset
from training.gesture.model import DualStreamEmotionTransformer
from training.gesture.train_gesture import build_windowed_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gesture model and dump JSON report.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--csv", type=Path, default=Path("data/emotion_dataset.csv"))
    parser.add_argument("--window-len", type=int, default=None, help="Override window length (default: ckpt max_len)")
    parser.add_argument("--window-stride", type=int, default=None, help="Override window stride (default: window_len)")
    parser.add_argument(
        "--windowed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the windowed dataframe (same as training). Disable to evaluate raw clips only.",
    )
    parser.add_argument(
        "--target-emotions",
        nargs="+",
        default=["angry", "disgusted", "fear", "happiness", "sadness", "surprised"],
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu override")
    parser.add_argument("--batch-size", type=int, default=512, help="Eval batch size (windows per step)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=True, help="Dataloader pin_memory"
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep workers alive between batches",
    )
    return parser.parse_args()


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Rebuild the model from a checkpoint and return it with its label maps and config."""
    # Safe loading; PosixPath stays allowed so checkpoints saved by older versions still load.
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
    return model, emo_to_idx, idx_to_emo, cfg


def summarize(report: List[dict]) -> dict:
    """Clip-level scores: one prediction per clip, from its averaged window logits."""
    y_true = [r["true_label"] for r in report]
    y_pred = [r["pred_label"] for r in report]
    # Top-2: the true emotion is among the clip's two most probable emotions (recall averaged per class).
    top2_recalls = []
    for label in sorted(set(y_true)):
        clips = [r for r in report if r["true_label"] == label]
        hits = sum(label in sorted(r["avg_probs"], key=r["avg_probs"].get, reverse=True)[:2] for r in clips)
        top2_recalls.append(hits / len(clips))
    return {
        "clips": len(report),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "top2_balanced_accuracy": sum(top2_recalls) / len(top2_recalls),
    }


class CachedGestureDataset(DanceEmotionDataset):
    """Dataset with JSON caching to avoid re-reading per window."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_cache = {}

    def _load_json(self, path: str):
        """Return a parsed JSON file, reading it from disk only once."""
        if path in self._json_cache:
            return self._json_cache[path]
        data = super()._load_json(path)
        self._json_cache[path] = data
        return data


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, emotion_to_idx, idx_to_emotion, cfg = load_checkpoint(args.ckpt, device)
    max_len = args.window_len or cfg["model"]["max_len"]
    stride = args.window_stride or max_len

    print(f"Using checkpoint: {args.ckpt} (max_len={max_len}, stride={stride}) on {device}")

    # Build dataframe: windowed (training-style) or raw clips
    if args.windowed:
        df = build_windowed_df(args.csv, max_len, stride, args.target_emotions)
    else:
        df = pd.read_csv(args.csv)
        df = df[df["emotion"].isin(args.target_emotions)].reset_index(drop=True)

    report = []
    total_rows = len(df)
    print(f"Building report for {total_rows} samples...")

    # Fast batched inference over the windowed dataframe
    ds = CachedGestureDataset(
        dataframe=df,
        emotion_to_idx=emotion_to_idx,
        max_len=max_len,
        visibility_threshold=0.5,
        replace_invalid_with_zeros=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        drop_last=False,
    )

    results = {}  # key: (pose_json, features_json, video_path) -> data

    # Subtracting the maximum first avoids overflow.
    def _softmax_np(logits: np.ndarray) -> np.ndarray:
        m = logits.max()
        exps = np.exp(logits - m)
        return exps / exps.sum()

    with torch.inference_mode():
        for batch_idx, (pose_seq, laban_seq, mask, labels, infos) in enumerate(loader, start=1):
            pose_seq = pose_seq.to(device, non_blocking=True)
            laban_seq = laban_seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(pose_seq, laban_seq, mask=mask)  # (B, C)
            logits_np = logits.cpu().numpy()

            # Default collate turns list of dicts into dict of lists; normalize to list of dicts.
            if isinstance(infos, dict):
                batch_infos = []
                keys = list(infos.keys())
                batch_size = len(next(iter(infos.values())))
                for i in range(batch_size):
                    batch_infos.append({k: infos[k][i] for k in keys})
            else:
                batch_infos = infos

            for i, info in enumerate(batch_infos):
                pose_path = info["pose_json"]
                feat_path = info["features_json"]
                video_path = info.get("video_path", "")
                start_idx = int(info.get("start_idx", 0))
                true_label = info["emotion"]

                key = (pose_path, feat_path, video_path)
                entry = results.setdefault(
                    key,
                    {
                        "video_path": video_path,
                        "pose_json": pose_path,
                        "features_json": feat_path,
                        "true_label": true_label,
                        "logits": [],
                    },
                )
                entry["logits"].append({"start_idx": start_idx, "logits": logits_np[i]})

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * args.batch_size} windows...")

    # Aggregate per clip
    for entry in results.values():
        window_logits = entry["logits"]
        logit_stack = np.stack([w["logits"] for w in window_logits], axis=0)
        avg_logits = logit_stack.mean(axis=0)
        avg_probs = _softmax_np(avg_logits)
        pred_idx = int(np.argmax(avg_logits))
        pred_label = idx_to_emotion[pred_idx]
        correct = pred_label == entry["true_label"]

        windows_out = []
        for w in window_logits:
            prob = _softmax_np(w["logits"])
            windows_out.append(
                {
                    "start_idx": int(w["start_idx"]),
                    "probs": {idx_to_emotion[i]: float(prob[i]) for i in range(len(prob))},
                }
            )

        report.append(
            {
                "video_path": entry["video_path"],
                "pose_json": entry["pose_json"],
                "features_json": entry["features_json"],
                "true_label": entry["true_label"],
                "pred_label": pred_label,
                "correct": bool(correct),
                "avg_probs": {idx_to_emotion[i]: float(avg_probs[i]) for i in range(len(avg_probs))},
                "windows": windows_out,
            }
        )

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or out_dir / f"training_predictions_{args.ckpt.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {len(report)} items to {out_path}")

    scores = summarize(report)
    print(
        f"Clip-level scores ({scores['clips']} clips): "
        f"accuracy {scores['accuracy'] * 100:.1f}% | macro F1 {scores['macro_f1'] * 100:.1f}% | "
        f"balanced accuracy {scores['balanced_accuracy'] * 100:.1f}% | "
        f"top-2 balanced accuracy {scores['top2_balanced_accuracy'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
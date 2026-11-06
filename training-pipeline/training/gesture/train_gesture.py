"""Train a dual-stream Transformer on windows of the extracted clips.

Each clip is cut into overlapping windows. Every window is one training sample, labeled with the
emotion of its clip. The result is a checkpoint.pt and a config.yaml in outputs/models/<model ID>/.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.gesture.gesture_dataset import DanceEmotionDataset
from training.gesture.model import DualStreamEmotionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gesture emotion model on GPU.")
    parser.add_argument("--config", type=Path, default=Path("configs/dualstream-large-w50-e100.yaml"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/emotion_dataset_train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/emotion_dataset_val.csv"))
    parser.add_argument("--test-csv", type=Path, default=None, help="Optional test CSV, only used to collect emotion labels; evaluate with eval_gesture.")
    parser.add_argument("--window-len", type=int, default=None, help="Clip length; defaults to config max_len.")
    parser.add_argument("--window-stride", type=int, default=None, help="Stride between windows (default: window_len).")
    parser.add_argument(
        "--target-emotions",
        nargs="+",
        default=["angry", "disgusted", "fear", "happiness", "sadness", "surprised"],
    )
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--use-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gesture-boost", type=float, default=2.0, help=">1 to upweight gesture segments.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Ignored if train/val CSVs exist.")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Read the YAML config. A missing file gives an empty config."""
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# (d_model, num_layers, nhead) of the named model sizes
MODEL_SIZES = {(256, 4, 8): "large", (128, 2, 4): "small"}


def model_id(model_cfg: dict, window_len: int, window_stride: int, num_epochs: int) -> str:
    """Model ID used as the default run name: dualstream-<size>-w<window>-s<stride>-e<epochs>."""
    shape = (model_cfg.get("d_model", 128), model_cfg.get("num_layers", 2), model_cfg.get("nhead", 4))
    size = MODEL_SIZES.get(shape, "d{}-l{}-h{}".format(*shape))
    return f"dualstream-{size}-w{window_len}-s{window_stride}-e{num_epochs}"


def build_windowed_df(data: Path | pd.DataFrame, window_len: int, window_stride: int, target_emotions: List[str]) -> pd.DataFrame:
    """Return one row per window of every clip, with its start index.

    The last window is aligned to the end of the clip, so no frames are lost. A clip shorter than a
    window gives one window, which the dataset pads.
    """
    if isinstance(data, Path):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    df = df[df["emotion"].isin(target_emotions)].reset_index(drop=True)

    def _segment_type(row):
        path_str = f"{row.get('video_path','')} {row['pose_json']} {row['features_json']}".lower()
        return "gesture" if "gesture" in path_str else "full"

    df["segment_type"] = df.apply(_segment_type, axis=1)

    def _count_frames(row):
        with open(row["pose_json"], "r") as f:
            pose = json.load(f)
        with open(row["features_json"], "r") as f:
            feat_frames = len(json.load(f)["features"])
        # frame_stride: every Nth video frame was used when the poses were extracted
        return min(len(pose["frames"]), feat_frames), int(pose.get("frame_stride", 1)), float(pose.get("fps", 0.0))

    expanded_rows = []
    for _, r in df.iterrows():
        total, frame_stride, fps = _count_frames(r)
        r["frame_stride"] = frame_stride
        r["sample_rate"] = fps / max(frame_stride, 1)
        if total <= 0:
            continue
        starts = list(range(0, max(total - window_len, 0) + 1, window_stride))
        if starts and starts[-1] + window_len < total:
            starts.append(total - window_len)
        if not starts:
            starts = [0]
        for s in starts:
            new_r = r.copy()
            new_r["start_idx"] = int(s)
            expanded_rows.append(new_r)

    return pd.DataFrame(expanded_rows)


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    emotion_to_idx: dict,
    max_len: int,
    batch_size: int,
    use_sampler: bool,
    gesture_boost: float,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    visibility_threshold: float,
):
    """Build the train and validation loaders and datasets.

    The optional sampler balances the emotions and gives gesture windows extra weight.
    """
    train_ds = DanceEmotionDataset(
        dataframe=train_df.reset_index(drop=True),
        emotion_to_idx=emotion_to_idx,
        max_len=max_len,
        visibility_threshold=visibility_threshold,
        replace_invalid_with_zeros=False,
    )
    val_ds = DanceEmotionDataset(
        dataframe=val_df.reset_index(drop=True),
        emotion_to_idx=emotion_to_idx,
        max_len=max_len,
        visibility_threshold=visibility_threshold,
        replace_invalid_with_zeros=False,
    )

    sampler = None
    if use_sampler:
        counts = Counter(train_df["emotion"])
        sample_weights = [1.0 / counts[row.emotion] for _, row in train_df.iterrows()]
        if "segment_type" in train_df.columns and gesture_boost != 1.0:
            boosted = []
            for w, (_, row) in zip(sample_weights, train_df.iterrows()):
                boosted.append(w * (gesture_boost if row.segment_type == "gesture" else 1.0))
            sample_weights = boosted
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    # PyTorch only accepts these with worker processes (num_workers > 0).
    if num_workers > 0:
        common_loader_kwargs.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, train_ds, val_ds


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    window_len = args.window_len or model_cfg.get("max_len", 30)
    # The model is built for window_len positions; save that, so loaders rebuild the same shape.
    cfg.setdefault("model", model_cfg)["max_len"] = window_len
    window_stride = args.window_stride or window_len
    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 10)
    learning_rate = args.learning_rate or float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_cfg.get("weight_decay", 0.0))

    # Build windowed train/val (and optional test) dataframes
    train_df_raw = pd.read_csv(args.train_csv)
    val_df_raw = pd.read_csv(args.val_csv) if args.val_csv and args.val_csv.exists() else None
    if val_df_raw is None or val_df_raw.empty:
        # fallback: split train_df_raw into train/val
        val_ratio = args.val_split
        train_df_raw, val_df_raw = train_test_split(
            train_df_raw,
            test_size=val_ratio,
            stratify=train_df_raw["emotion"],
            random_state=args.seed,
        )

    train_df = build_windowed_df(train_df_raw, window_len, window_stride, args.target_emotions)
    val_df = build_windowed_df(val_df_raw, window_len, window_stride, args.target_emotions)

    if train_df.empty:
        raise SystemExit("No train samples found after windowing.")
    if val_df.empty:
        raise SystemExit("No val samples found after windowing.")

    # The model must see videos sampled the same way at inference, so all clips must agree.
    frame_strides = set(train_df["frame_stride"]) | set(val_df["frame_stride"])
    if len(frame_strides) != 1:
        raise SystemExit(
            f"Pose files were extracted with different frame strides {sorted(frame_strides)}; "
            "re-run the pipeline with a single --stride."
        )
    frame_stride = int(frame_strides.pop())
    # Analyzed frames per second. The dashboard samples new videos at this rate, whatever their frame rate.
    sample_rate = float(train_df["sample_rate"].median())

    # If test CSV provided, build windowed test for optional eval after training
    test_df = None
    if args.test_csv:
        test_raw = pd.read_csv(args.test_csv)
        test_df = build_windowed_df(test_raw, window_len, window_stride, args.target_emotions)
        if test_df.empty:
            print("Warning: test CSV provided but windowed test is empty.")

    all_for_vocab = pd.concat([train_df, val_df] + ([test_df] if test_df is not None else []))
    # Sorted, so the label indices don't depend on the order of the rows.
    emotion_names = sorted(all_for_vocab["emotion"].dropna().astype(str).unique())
    emotion_to_idx = {emo: i for i, emo in enumerate(emotion_names)}
    idx_to_emotion = {idx: emo for emo, idx in emotion_to_idx.items()}

    train_loader, val_loader, train_ds, val_ds = make_loaders(
        train_df=train_df,
        val_df=val_df,
        emotion_to_idx=emotion_to_idx,
        max_len=window_len,
        batch_size=batch_size,
        use_sampler=args.use_sampler,
        gesture_boost=args.gesture_boost,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        visibility_threshold=args.visibility_threshold,
    )

    model = DualStreamEmotionTransformer(
        pose_dim=33 * 3,
        laban_dim=6,
        num_emotions=len(emotion_to_idx),
        d_model=model_cfg.get("d_model", 128),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 2),
        max_len=window_len,
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    torch.backends.cudnn.benchmark = True

    print(
        f"Starting training on {device} | epochs={num_epochs} | batch_size={batch_size} | "
        f"workers={args.num_workers} | amp={args.amp} | sampler={args.use_sampler}"
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        t0 = time.time()

        for pose_seq, laban_seq, mask, labels, _info in train_loader:
            pose_seq = pose_seq.to(device, non_blocking=True)
            laban_seq = laban_seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(pose_seq, laban_seq, mask=mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)

        avg_train_loss = total_loss / max(1, total_seen)

        # This accuracy is per window. Final scores are per clip (see eval_gesture.py) and usually higher.
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for pose_seq, laban_seq, mask, labels, _info in val_loader:
                pose_seq = pose_seq.to(device, non_blocking=True)
                laban_seq = laban_seq.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=args.amp):
                    logits = model(pose_seq, laban_seq, mask=mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(1, total)
        dt = time.time() - t0
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} - train_loss={avg_train_loss:.4f} "
            f"- val_acc={val_acc:.3f} - time={dt:.1f}s"
        )

    # Save checkpoint, named after the model ID unless --run-name is given; never overwrite an earlier run
    run_name = args.run_name or model_id(model_cfg, window_len, window_stride, num_epochs)
    if not args.run_name and (args.save_dir / run_name).exists():
        run_name = f"{run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = args.save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Inference-only checkpoint: no optimizer state (training is never resumed, and it
    # would triple the file size). Paths are stored as strings so the file loads with
    # weights_only=True on any OS.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cfg": cfg,
        "emotion_to_idx": emotion_to_idx,
        "idx_to_emotion": idx_to_emotion,
        "frame_stride": frame_stride,
        "sample_rate": sample_rate,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    torch.save(checkpoint, save_dir / "checkpoint.pt")
    print(f"Saved checkpoint to {save_dir / 'checkpoint.pt'}")

    # Persist a run config with window/stride and training metadata alongside the checkpoint
    run_meta = {
        "frame_stride": frame_stride,
        "sample_rate": sample_rate,
        "window_len": window_len,
        "window_stride": window_stride,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_sampler": args.use_sampler,
        "gesture_boost": args.gesture_boost,
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "test_csv": str(args.test_csv) if args.test_csv else None,
    }
    cfg_out = dict(cfg)
    cfg_out["run_meta"] = run_meta
    with open(save_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_out, f)
    print(f"Wrote config with run_meta to {save_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
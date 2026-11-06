"""PyTorch dataset that serves fixed-length windows of a clip's pose and LMA files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class DanceEmotionDataset(Dataset):
    """
    Loads pose & Laban JSONs, aligns frames, slices/pads to max_len, and
    returns (pose_seq, laban_seq, mask, label, info).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        emotion_to_idx: Dict[str, int],
        max_len: int = 64,
        visibility_threshold: float = 0.5,
        replace_invalid_with_zeros: bool = True,
    ):
        """Wrap a dataframe with pose_json, features_json and emotion columns.

        An optional start_idx column selects the window of each row. With replace_invalid_with_zeros,
        joints under visibility_threshold become (0, 0, 0). Training, evaluation and the dashboard all
        keep the original coordinates, so there the threshold only affects the counts in `info`.
        """
        self.df = dataframe.reset_index(drop=True)
        self.emotion_to_idx = emotion_to_idx
        self.max_len = max_len
        self.visibility_threshold = visibility_threshold
        self.replace_invalid_with_zeros = replace_invalid_with_zeros

        self.J = 33  # Mediapipe joints
        self.F = 6   # Laban features

    def __len__(self) -> int:
        return len(self.df)

    def _load_json(self, path: str):
        """Read a JSON file. eval_gesture.py overrides this to cache the files."""
        with open(path, "r") as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Return one window, padded with zeros up to max_len. The mask is True on padded steps."""
        row = self.df.iloc[idx]

        pose_data = self._load_json(row["pose_json"])
        feat_data = self._load_json(row["features_json"])

        # frame_index -> laban vector
        laban_by_frame: Dict[int, torch.Tensor] = {}
        for f in feat_data["features"]:
            laban_by_frame[f["frame_index"]] = torch.tensor(
                [
                    f["active_ratio"],
                    f["body_expansion"],
                    f["vertical_level"],
                    f["hand_energy"],
                    f["foot_energy"],
                    f["torso_inclination"],
                ],
                dtype=torch.float32,
            )

        pose_seq: List[List[List[float]]] = []
        laban_seq: List[torch.Tensor] = []
        num_low_vis_joints = 0
        num_joints_zeroed = 0

        for frame in pose_data["frames"]:
            # Keep the frames present in both files, so the two streams stay aligned.
            fi = frame["frame_index"]
            if fi not in laban_by_frame:
                continue

            joints = []
            for lm in frame["landmarks_world"]:
                vis = lm["visibility"]
                if vis < self.visibility_threshold:
                    num_low_vis_joints += 1
                    if self.replace_invalid_with_zeros:
                        joints.append([0.0, 0.0, 0.0])
                        num_joints_zeroed += 1
                    else:
                        joints.append([lm["x"], lm["y"], lm["z"]])
                else:
                    joints.append([lm["x"], lm["y"], lm["z"]])

            assert len(joints) == self.J
            pose_seq.append(joints)
            laban_seq.append(laban_by_frame[fi])

        T_raw = len(pose_seq)
        start_idx = int(row.get("start_idx", 0) or 0)
        start_idx = max(start_idx, 0)
        end_idx = start_idx + self.max_len
        pose_seq = pose_seq[start_idx:end_idx]
        laban_seq = laban_seq[start_idx:end_idx]

        T_window = len(pose_seq)
        pose_seq_t = torch.tensor(pose_seq, dtype=torch.float32)
        laban_seq_t = torch.stack(laban_seq, dim=0) if laban_seq else torch.zeros(0, self.F)

        frames_truncated = 0
        if T_window > self.max_len:
            frames_truncated = T_window - self.max_len
            pose_seq_t = pose_seq_t[: self.max_len]
            laban_seq_t = laban_seq_t[: self.max_len]
            T_used = self.max_len
        else:
            T_used = T_window

        pad_len = self.max_len - T_used
        if pad_len > 0:
            pose_pad = torch.zeros(pad_len, self.J, 3)
            laban_pad = torch.zeros(pad_len, self.F)
            pose_seq_t = torch.cat([pose_seq_t, pose_pad], dim=0)
            laban_seq_t = torch.cat([laban_seq_t, laban_pad], dim=0)

        mask = torch.zeros(self.max_len, dtype=torch.bool)
        if pad_len > 0:
            mask[T_used:] = True

        emotion_str = row["emotion"]
        label = self.emotion_to_idx[emotion_str]

        info = {
            "start_idx": start_idx,
            "window_len": self.max_len,
            "video_path": row.get("video_path", ""),
            "pose_json": row["pose_json"],
            "features_json": row["features_json"],
            "emotion": emotion_str,
            "T_raw": T_raw,
            "T_used": T_used,
            "frames_truncated": frames_truncated,
            "num_low_vis_joints": num_low_vis_joints,
            "num_joints_zeroed": num_joints_zeroed,
        }

        return pose_seq_t, laban_seq_t, mask, label, info
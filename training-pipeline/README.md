# Training Pipeline

Turn dance videos into 3D poses and movement features, build a dataset, then train and evaluate emotion recognition models.

- [Requirements](#requirements)
- [Installation](#installation)
- [1. Prepare the videos](#1-prepare-the-videos)
- [2. Extract poses and features](#2-extract-poses-and-features)
- [3. Build the dataset](#3-build-the-dataset)
- [4. Train a model](#4-train-a-model)
- [5. Evaluate it](#5-evaluate-it)
- [6. Use it in the dashboard](#6-use-it-in-the-dashboard)
- [Configurations](#configurations)
- [Output layout](#output-layout)
- [Training time](#training-time)
- [Troubleshooting](#troubleshooting)

## Requirements

- **Python 3.10, 3.11 or 3.12**
- **[FFmpeg](https://ffmpeg.org/download.html)**, only to convert `.ts` videos (step 1)
- A CUDA GPU is recommended for training, but not required: without one, training runs on the CPU, more slowly (see [Training time](#training-time))

**Run every command from this `training-pipeline/` folder.**

## Installation

Run these commands once, from the `training-pipeline/` folder.

```bash
python3 -m venv .venv              # Windows: python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Prepare the videos

Put your videos in `training-pipeline/assets/`. The emotion label is the **first word of the file name**, so name each file `<emotion>_<anything>.<ext>`, where `<emotion>` is one of:

`angry` · `disgusted` · `fear` · `happiness` · `sadness` · `surprised`

For example `happiness_12_gesture.avi` or `fear_03_improvisation.mp4`. If a clip is a short gesture rather than a full performance, put `gesture` anywhere in its name: the training sampler shows these clips more often (see `--gesture-boost`).

The pipeline reads `.avi`, `.mp4`, `.mov` and `.mkv` files. Other files are listed as ignored when the run starts.

**Optional helpers.** They were written for the dataset this project was built on, which was partly recorded as `.ts` files and named with the category first. Use them only if your videos are in the same situation:

```bash
# Your videos are .ts files, which OpenCV decodes unreliably: convert X.ts to X_ts.avi with FFmpeg.
# Each .ts file is deleted once its conversion succeeds, so keep a copy of your originals.
python src/convert_ts_to_avi.py            # dry run: shows what would be converted
python src/convert_ts_to_avi.py --apply    # add --overwrite to redo existing files

# Your .avi files are named gesture_<number>_<emotion>.avi or improvisation_<number>_<emotion>.avi:
# rename gesture_12_happiness.avi to happiness_12_gesture.avi
python src/rename_assets.py                # dry run
python src/rename_assets.py --apply
```

The model can only learn from what the pose tracker sees, so the quality of your videos matters more than any setting:

| | Use | Otherwise |
|---|---|---|
| **People** | One person in the frame | The tracker follows a single person and can jump between several people, mixing their movements. Nothing detects this, so these clips silently corrupt the training data |
| **Framing** | The whole body in view, and not too small in the frame | Missing limbs lower the quality of the pose, and a very small person may not be detected at all |
| **Image** | Steady camera, good lighting, sharp image | The pose gets noisier, often without any error |
| **Length** | At least one window: 4 seconds with the default settings at 50 fps | Shorter clips are kept, but padded with empty frames. Above `--max-frames`, the rest of the video is ignored and the run prints a note |
| **Frame rate** | The same for all your clips | The model learns at one sampling rate. It is saved in the checkpoint, and the dashboard samples new videos at that rate |

Check a few clips before you extract a large dataset. `--keep-frames` saves the images that this check shows next to the 3D skeleton:

```bash
python src/run_full_pipeline.py --videos <video-file> --keep-frames
python src/visualize_pose_multiview.py outputs/batch_runs/<clip>/run_blended/poses/poses.json
```

`<video-file>` is the name of a video in `assets/`, for example `happiness_12_gesture.avi`, and `<clip>` is the same name without the extension.

## 2. Extract poses and features

```bash
python src/run_full_pipeline.py
```

For each video in `assets/`, the script:

1. Samples one frame in four (`--stride 4`).
2. Runs MediaPipe Pose (heaviest model) on four versions of each frame:

   | Pass | Preprocessing |
   |---|---|
   | `raw` | none |
   | `segmentation` | person segmentation, crop around the person, background blur |
   | `denoise` | bilateral filter and unsharp mask, then the segmentation crop |
   | `illumination` | CLAHE contrast and gamma correction, then the segmentation crop |

3. Blends the passes: for each frame, keeps the pass with the highest ratio of visible 3D landmarks.
4. Applies quality control: drops frames where fewer than 60% of the landmarks are usable, fills gaps of up to 5 samples by linear interpolation, and clamps joints that jump more than 0.38 m between two samples.
5. Computes six LMA-inspired features per frame:

   | Feature | Meaning |
   |---|---|
   | `active_ratio` | Share of the 33 landmarks detected |
   | `body_expansion` | Volume of the 3D bounding box around the body (m³) |
   | `vertical_level` | Height of the hips |
   | `hand_energy` | Mean speed of the hand landmarks (m/s) |
   | `foot_energy` | Mean speed of the foot landmarks (m/s) |
   | `torso_inclination` | Tilt of the hip-to-shoulder axis |

A video that cannot be processed, because it is unreadable or shows no person, does not stop the run: the others continue, and the failed ones are listed at the end. Finished videos are skipped if you run it again, so you can fix or remove a file and resume.

It writes `outputs/run_blended_qc/<clip>_poses.json` and `<clip>_features.json`, and keeps the per-pass results in `outputs/batch_runs/<clip>/`. Videos that already have both final files are skipped.

The per-pass results take about 0.7 MB of disk per second of video. Only the final files are used afterwards, so you can delete `outputs/batch_runs/` when the extraction is done.

| Option | Default | Description |
|---|---|---|
| `--assets-dir` | `assets` | Folder with the input videos |
| `--videos` | all | Process only these videos (paths relative to `--assets-dir`) |
| `--stride` | `4` | Analyze one video frame in N |
| `--max-frames` | `3000` | Maximum sampled frames per video (3000 × 4 = 12,000 frames, 4 minutes at 50 fps) |
| `--keep-frames` | off | Also save the sampled frames as images, for the visual check in step 1. Needs about 12 MB of disk per second of video, so use it only on the clips you want to inspect |
| `--reprocess-existing` | off | Re-run videos that already have final outputs |
| `--final-output-dir` | `outputs/run_blended_qc` | Where the final pose and feature files go |
| `--intermediate-root` | `outputs/batch_runs` | Where the per-pass results go |

> **Keep the same `--stride` for all your videos**, ideally the default 4, used for the models of this project. The frame stride is recorded in the pose files, training refuses data that mixes several strides, and it is saved in the trained model so the dashboard samples videos the same way.

## 3. Build the dataset

```bash
python src/tools/build_emotion_dataset.py
```

This indexes every pose and feature pair in `outputs/run_blended_qc/` and writes `data/emotion_dataset.csv`, plus a train, validation and test split (70 / 15 / 15, stratified by emotion) in `data/emotion_dataset_{train,val,test}.csv`.

| Option | Default | Description |
|---|---|---|
| `--val-ratio` | `0.15` | Share of clips for validation |
| `--test-ratio` | `0.15` | Share of clips for testing |
| `--seed` | `42` | Random seed of the split |
| `--qc-dir` | `outputs/run_blended_qc` | Folder with the pose and feature files |

With the default ratios, each emotion needs **at least 7 clips** to be split, and many more to train a useful model: the dataset of this project had 29 to 46 per emotion. A file name with an unknown emotion, such as `happy_01.mp4`, is reported with a warning, and training ignores that clip.

> **Several clips of the same performance?** The split is made per clip. If you filmed a performance from several viewpoints, or cut it into several clips, the same performance can land in both training and test, which inflates the test scores. Keep such clips in the same split.

## 4. Train a model

```bash
python -m training.gesture.train_gesture \
  --config configs/<config>.yaml \
  --window-len <window> --window-stride <stride> \
  --test-csv data/emotion_dataset_test.csv
```

- `<config>`: a configuration from [Configurations](#configurations), for example `dualstream-large-w50-e100`
- `<window>`: samples per window, the `w` number of the config
- `<stride>`: samples between two windows

**Example: the settings of the best model of this project**, `dualstream-large-w50-s10-e100`:

```bash
python -m training.gesture.train_gesture \
  --config configs/dualstream-large-w50-e100.yaml \
  --window-len 50 --window-stride 10 \
  --test-csv data/emotion_dataset_test.csv
```

Each training example is a window of `--window-len` consecutive samples, taken every `--window-stride` samples along each clip. The model and training settings come from the config; command-line options override them.

| Option | Default | Description |
|---|---|---|
| `--config` | `configs/dualstream-large-w50-e100.yaml` | Model and training configuration |
| `--train-csv`, `--val-csv` | `data/emotion_dataset_{train,val}.csv` | Training and validation splits |
| `--test-csv` | none | Only used to collect the emotion labels; evaluate with step 5 |
| `--window-len` | config `max_len` | Samples per window |
| `--window-stride` | `--window-len` | Step between two windows |
| `--batch-size`, `--num-epochs`, `--learning-rate`, `--weight-decay` | from config | Override the config |
| `--use-sampler` / `--no-use-sampler` | on | Balance emotions with a weighted random sampler |
| `--gesture-boost` | `2.0` | Extra sampling weight for gesture clips |
| `--amp` / `--no-amp` | on | Mixed precision on CUDA GPUs |
| `--num-workers` | `8` | Data-loading processes (`0` loads in the main process) |
| `--save-dir` | `outputs/models` | Where trained models are saved |
| `--run-name` | model ID | Name of the model folder (default: the [model ID](#configurations), plus a timestamp if that folder already exists) |
| `--seed` | `42` | Random seed |

The model is saved in a new folder, `outputs/models/<model-folder>/`. By default the folder name is the [model ID](#configurations), `dualstream-<size>-w<window>-s<stride>-e<epochs>`, and the end of the training output gives its path:

```text
Saved checkpoint to outputs/models/dualstream-large-w50-s10-e100/checkpoint.pt
```

The folder contains:

- `checkpoint.pt`: weights, configuration, emotion labels and the frame stride of the training data (no optimizer state, so it is about a third of a full training checkpoint);
- `config.yaml`: the configuration plus the run settings.

Training again with the same settings never overwrites a model: the new folder gets a timestamp, for example `dualstream-large-w50-s10-e100-20261029_153012`.

## 5. Evaluate it

```bash
python -m training.gesture.eval_gesture \
  --ckpt outputs/models/<model-folder>/checkpoint.pt \
  --csv data/emotion_dataset_test.csv \
  --window-len <window> --window-stride <stride> \
  --output outputs/reports/<model-folder>_test.json
```

Use the same `<window>` and `<stride>` as for training. The script averages the window predictions of each clip, prints the clip-level scores, and saves every clip's prediction and per-window probabilities to the `--output` JSON.

**Example: the model trained in the example of step 4:**

```bash
python -m training.gesture.eval_gesture \
  --ckpt outputs/models/dualstream-large-w50-s10-e100/checkpoint.pt \
  --csv data/emotion_dataset_test.csv \
  --window-len 50 --window-stride 10 \
  --output outputs/reports/dualstream-large-w50-s10-e100_test.json
```

The output ends with the clip-level scores. For the best model of this project, on the test split of its 206 clips:

```text
Clip-level scores (31 clips): accuracy 71.0% | macro F1 69.7% | balanced accuracy 69.6% | top-2 balanced accuracy 78.1%
```

Your scores depend on your dataset and its split, so they can differ from these.

## 6. Use it in the dashboard

Copy the model folder into `dashboard/models/`, then refresh the dashboard page. The model appears in the model list in the sidebar:

```bash
cp -r outputs/models/<model-folder> ../dashboard/models/
```

For example, the model trained in the example of step 4:

```bash
cp -r outputs/models/dualstream-large-w50-s10-e100 ../dashboard/models/
```

If `dashboard/models/` already has a folder with that name, give the copy a new name, or it replaces the files in that folder.

## Configurations

Every config in [`configs/`](configs/) follows `dualstream-<size>-w<window>-e<epochs>.yaml`. The window stride is chosen at training time, which gives the model ID: `dualstream-large-w50-e100.yaml` trained with `--window-stride 10` becomes `dualstream-large-w50-s10-e100`.

| Size | `d_model` | Encoder layers per stream | Attention heads | Dropout |
|---|---|---|---|---|
| `small` | 128 | 2 | 4 | 0.10 |
| `large` | 256 | 4 | 8 | 0.15 |

| Config | Window | Epochs | Batch size | Learning rate | Weight decay |
|---|---|---|---|---|---|
| `dualstream-large-w50-e100` (best model) | 50 | 100 | 256 | 3e-4 | 1e-2 |
| `dualstream-large-w60-e100` | 60 | 100 | 256 | 3e-4 | 1e-2 |
| `dualstream-large-w60-e70` | 60 | 70 | 256 | 3e-4 | 1e-2 |
| `dualstream-large-w50-e40` | 50 | 40 | 256 | 3e-4 | 1e-2 |
| `dualstream-large-w48-e70` | 48 | 70 | 256 | 3e-4 | 1e-2 |
| `dualstream-large-w48-e40` | 48 | 40 | 256 | 3e-4 | 1e-2 |
| `dualstream-small-w60-e100` | 60 | 100 | 256 | 3e-4 | 1e-2 |
| `dualstream-small-w50-e100` | 50 | 100 | 256 | 3e-4 | 1e-2 |
| `dualstream-small-w64-e20` | 64 | 20 | 8 | 1e-3 | 1e-4 |
| `dualstream-small-w10-e25` | 10 | 25 | 8 | 1e-3 | 0 |

The test scores of the models trained from these configs are in the [main README](../README.md#results).

## Output layout

```
training-pipeline/
├── assets/                    # your input videos (not versioned)
├── data/                      # dataset CSVs (not versioned)
└── outputs/                   # everything generated (not versioned)
    ├── batch_runs/<clip>/     # per-pass pose estimation results
    ├── run_blended_qc/        # final <clip>_poses.json and <clip>_features.json
    ├── models/<model-folder>/ # checkpoint.pt and config.yaml
    └── reports/               # evaluation JSON files
```

These three folders stay on your computer: git ignores their contents, so your videos and the data extracted from them can't end up in a commit by mistake.

## Training time

Average time per epoch, over 40 epochs, for a `large` model with windows of 60 and a stride of 10, with the same batch size on every device:

| Machine | GPU | CPU | RAM | Average time per epoch (s) |
|---|---|---|---|---|
| Apple M1 Pro laptop | not used | 8 cores | 16 GB, shared with the GPU | 570 |
| Cloud instance | 1 × NVIDIA RTX 4000 Ada, 20 GB | 8 vCPUs | 32 GB | 249 |
| Cloud instance | 1 × NVIDIA H100, 80 GB | 20 vCPUs | 240 GB | 91 |

These times were measured while training the models of this project. They are not benchmarks of the hardware. On your computer, training can be faster or slower, depending on its configuration and on what else is running on it.

## Troubleshooting

**A video is listed at the end of the extraction as not processed**
"No person was detected" or "the video could not be read": check that the file plays and that one person is clearly visible, then fix or remove it and run the same command again.

**`Not enough clips to split the dataset`**
Add clips for the emotions with the fewest, or change `--val-ratio` and `--test-ratio`.

**`Pose files were extracted with different frame strides`**
Some clips were extracted with another `--stride`. Re-extract them with one stride: `python src/run_full_pipeline.py --stride 4 --reprocess-existing`.

**Training without a CUDA GPU**
Add `--no-amp` (mixed precision only works on CUDA), and lower `--batch-size` if you run out of memory.

**Data-loading errors on macOS or Windows**
Try `--num-workers 0`, which loads data in the main process.

**`ffmpeg not found on PATH`**
Install FFmpeg (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian or Ubuntu, or the [official builds](https://ffmpeg.org/download.html) on Windows). It is only needed by `convert_ts_to_avi.py`.

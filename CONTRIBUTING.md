# Contributing

Thanks for your interest in this project. Bug reports and questions are welcome. This repository does not accept pull requests, but you are free to fork it and change your own copy under the [MIT License](LICENSE).

## Reporting a bug or asking a question

Open an issue on GitHub and include:

- what you ran (the exact command, or what you did in the dashboard),
- what you expected and what happened instead, with the full error message,
- your operating system, Python version (`python --version`) and whether you use a GPU.

**Never attach videos of people, or pose and feature files extracted from them**, unless you have their explicit consent to share them publicly.

## Development setup

```bash
git clone https://github.com/TaoufiQ-TbT/emotion-recognition-from-movement.git
cd emotion-recognition-from-movement
python3 -m venv .venv              # Windows: python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r training-pipeline/requirements.txt -r dashboard/requirements.txt
```

You also need [FFmpeg](https://ffmpeg.org/download.html) on your `PATH`. Supported Python versions are 3.10 to 3.12.

## How the code is organized

| Folder | What it contains |
|---|---|
| `training-pipeline/src/` | Preprocessing, pose estimation, quality control and LMA feature extraction |
| `training-pipeline/training/` | Dataset, model, training and evaluation |
| `training-pipeline/configs/` | Model and training configurations |
| `dashboard/` | The web application (`app.py`, `jobs.py`, `pipeline.py`) |
| `dashboard/src/`, `dashboard/training/` | **Copies** of `training-pipeline/src/` and `training-pipeline/training/`, so the dashboard runs on its own |
| `scripts/` | Maintenance scripts |

### Keep the dashboard copy in sync

Always edit the shared code in `training-pipeline/`, never in `dashboard/src/` or `dashboard/training/`. Then copy your changes over:

```bash
python scripts/sync_dashboard.py           # copy changed files into dashboard/
python scripts/sync_dashboard.py --check   # verify; exits with code 1 if out of sync
```

## Naming conventions

Configs and models use one technical ID scheme:

| Kind | Pattern | Example |
|---|---|---|
| Config | `dualstream-<size>-w<window>-e<epochs>.yaml` | `dualstream-large-w50-e100.yaml` |
| Trained model | `dualstream-<size>-w<window>-s<stride>-e<epochs>` | `dualstream-large-w50-s10-e100` |

`size` is `small` (`d_model` 128, 2 layers, 4 heads) or `large` (`d_model` 256, 4 layers, 8 heads). `window` is the number of pose frames per window, and `stride` the window stride passed with `--window-stride` at training time.

## If you change the code in your copy

1. Match the style of the surrounding code (type hints, docstrings, `pathlib` paths).
2. Run `python scripts/sync_dashboard.py --check`.
3. Run what you changed. For training pipeline changes, process at least one video; for dashboard changes, upload a video and check the predictions appear.
4. Update the relevant README when you change a command, an option or a default.

## Data and privacy

The `.gitignore` blocks videos, extracted poses and features, and trained model weights. Do not force-add them (`git add -f`), and never publish recordings of people without their explicit consent.

## License

The code is available under the [MIT License](LICENSE). If you share a modified copy, keep the copyright and license notice.

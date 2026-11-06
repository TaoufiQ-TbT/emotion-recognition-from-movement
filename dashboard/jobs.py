"""Background analysis jobs for the dashboard.

Each upload becomes a Job running on its own thread, so the web server stays responsive while the
page polls for progress. Jobs live in memory: restarting the dashboard forgets them, but their
files stay on disk.
"""
from __future__ import annotations

import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from werkzeug.utils import secure_filename

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from pipeline import (
    DEFAULT_SAMPLE_RATE,
    InferenceResult,
    convert_video_to_mp4,
    get_model_config,
    run_inference_from_features,
    run_multi_pass_inference,
)


@dataclass
class Job:
    """One analysis: its files, its settings and its progress (queued, processing, ready or error)."""
    job_id: str
    status: str
    message: str
    video_filename: str
    video_path: Path
    model_path: Path
    run_dir: Path
    result: Optional[InferenceResult] = None
    error: Optional[str] = None
    logs: list[str] = field(default_factory=list)
    sample_rate: float = DEFAULT_SAMPLE_RATE
    max_frames: Optional[int] = None
    preprocess_mode: str = "segmentation"

    @property
    def media_url(self) -> str:
        """URL of the playable video."""
        return f"/media/{self.job_id}/{self.video_filename}"


class JobStore:
    """Thread-safe list of the analysis jobs."""
    def __init__(self, uploads_root: Path, runs_root: Path) -> None:
        self.uploads_root = uploads_root
        self.runs_root = runs_root
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job_from_path(
        self,
        *,
        filename: str,
        existing_path: Path,
        model_path: Path,
        sample_rate: float,
        max_frames: Optional[int],
        preprocess_mode: str,
    ) -> Job:
        """Create a job from a file that is already on disk (e.g. via chunked upload)."""
        # The name comes from the browser: keep only a safe basename.
        filename = secure_filename(filename) or "video.mp4"
        job_id = uuid.uuid4().hex[:10]
        upload_dir = self.uploads_root / job_id
        run_dir = self.runs_root / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        target_path = upload_dir / filename
        if existing_path.resolve() != target_path.resolve():
            import shutil
            shutil.move(str(existing_path), str(target_path))

        job = Job(
            job_id=job_id,
            status="queued",
            message="Queued",
            video_filename=filename,
            video_path=target_path,
            model_path=model_path,
            run_dir=run_dir,
            sample_rate=sample_rate,
            max_frames=max_frames,
            preprocess_mode=preprocess_mode,
        )

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job

    def _update_job(self, job_id: str, **updates) -> None:
        """Change job fields under the lock, so readers never see a half-done update."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in updates.items():
                setattr(job, key, value)

    def _run_job(self, job_id: str) -> None:
        """Worker thread: convert the video, then run the full analysis."""
        job = self.get(job_id)
        if not job:
            return

        self._update_job(job_id, status="processing", message="Converting video...")

        try:
            # Browsers can't play every format, so the player and the analysis both use this MP4.
            mp4_path = convert_video_to_mp4(job.video_path)

            job.video_path = mp4_path
            job.video_filename = mp4_path.name

            self._update_job(job_id, video_path=mp4_path, video_filename=mp4_path.name)

            result = run_multi_pass_inference(
                mp4_path,
                ckpt=job.model_path,
                output_dir=job.run_dir,
                sample_rate=job.sample_rate,
                max_frames=job.max_frames,
                progress=lambda message: self._update_job(job_id, message=message),
            )
            self._update_job(
                job_id,
                status="ready",
                message="Done",
                result=result,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - show full error to user
            self._update_job(
                job_id,
                status="error",
                message="Failed",
                error=str(exc),
            )

    def reprocess_job(self, job_id: str, new_model_path: Path) -> bool:
        """Trigger re-inference on an existing job if features persist."""
        job = self.get(job_id)
        if not job:
            return False

        pose_json = job.run_dir / "poses.json"
        features_json = job.run_dir / "features.json"

        if not (pose_json.exists() and features_json.exists()):
            return False

        # Existing poses only fit a model trained at the same sampling rate; otherwise redo the analysis.
        new_sample_rate = get_model_config(new_model_path).get("sample_rate", DEFAULT_SAMPLE_RATE)
        if new_sample_rate != job.sample_rate:
            self._update_job(job_id, model_path=new_model_path, sample_rate=new_sample_rate, result=None)
            threading.Thread(target=self._run_job, args=(job_id,), daemon=True).start()
            return True

        self._update_job(job_id, model_path=new_model_path, status="processing", message="Switching Model...", result=None)

        thread = threading.Thread(target=self._run_reprocess, args=(job_id, pose_json, features_json), daemon=True)
        thread.start()
        return True

    def _run_reprocess(self, job_id: str, pose_json: Path, features_json: Path) -> None:
        """Worker thread: run only the model, on the saved poses and features."""
        job = self.get(job_id)
        if not job:
            return

        try:
            result = run_inference_from_features(
                video_path=job.video_path,
                pose_json=pose_json,
                features_json=features_json,
                ckpt=job.model_path,
                output_dir=job.run_dir,
            )

            self._update_job(
                job_id,
                status="ready",
                message="Done",
                result=result,
                error=None,
            )
        except Exception as exc:
             self._update_job(
                job_id,
                status="error",
                message="Reprocess Failed",
                error=str(exc),
            )

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

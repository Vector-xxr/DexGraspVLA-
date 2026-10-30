"""Rollout video writing for AR5 evaluations.

Deliberately not reusing ``aloha_utils.save_rollout_video``: that module imports
``real_env``, which imports ``interbotix_xs_modules`` -- a ROS package. This
setup is SDK-only, so the ALOHA import chain must not be pulled in.
"""

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _slugify(text: str, max_length: int = 60) -> str:
    """Filesystem-safe fragment of a task description."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text.strip().lower()]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_length] or "task"


def save_rollout_video(
    frames: List[np.ndarray],
    episode_idx: int,
    success: bool,
    task_description: str,
    fps: int = 20,
    out_dir: str = "./experiments/videos",
    notes: Optional[str] = None,
    log_file=None,
) -> Optional[str]:
    """Write one episode's frames to an MP4. Returns the path, or None if empty.

    Frames are RGB uint8 as produced by the camera drivers.
    """
    if not frames:
        logger.warning(f"Episode {episode_idx}: no frames captured, skipping video")
        return None

    os.makedirs(out_dir, exist_ok=True)
    name = f"episode_{episode_idx:03d}_{'success' if success else 'failure'}_{_slugify(task_description)}"
    if notes:
        name += f"_{_slugify(notes, 20)}"
    path = os.path.join(out_dir, name + ".mp4")

    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        with imageio.get_writer(path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
    except Exception as exc:  # noqa: BLE001
        # A failed video write must never cost an episode's success record.
        logger.error(f"Could not write {path}: {exc}")
        return None

    message = f"Saved rollout video to {path}"
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()
    return path

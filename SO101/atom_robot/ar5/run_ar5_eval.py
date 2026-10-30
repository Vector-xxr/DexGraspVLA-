"""
run_ar5_eval.py

Evaluates a policy on a real ROKAE AR5 arm with an Inspire dexterous hand.

Adapted from ``run_aloha_eval.py``. Two cameras instead of three, a single arm,
and a 7-dim delta end-effector action space.

VEL lives entirely on the server: whether actions are energy-corrected is
decided by the ``--refinement_method`` flag passed to ``deploy.py``. Nothing in
this file changes between the baseline and the VEL condition, which is what
makes the A/B comparison a clean paired experiment.

    # baseline
    VLA_ROBOT_PLATFORM=AR5 python vla-scripts/deploy.py \
        --pretrained_checkpoint <ckpt> --refinement_method none

    # VEL
    VLA_ROBOT_PLATFORM=AR5 python vla-scripts/deploy.py \
        --pretrained_checkpoint <ckpt> --refinement_method vel \
        --energy_checkpoint_path <energy_head.pt> --energy_alpha 0.1 --energy_k 1

    # client, unchanged in both cases
    python atom_robot/ar5/run_ar5_eval.py --vla_server_url 127.0.0.1
"""

# Must be the first heavyweight import in the process -- see the module
# docstring of prismatic/_import_order.py. Importing transformers or TensorFlow
# before torchvision segfaults the interpreter with no traceback.
import prismatic._import_order  # noqa: F401  -- import order matters

import logging
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm

sys.path.append(".")
from atom_robot.ar5.ar5_env import AR5Env  # noqa: E402
from atom_robot.ar5.config import AR5Config  # noqa: E402
from atom_robot.ar5.video import save_rollout_video  # noqa: E402

# **这一个脚本仍然要 VEL 仓库。** 它跑的是 OpenVLA 策略评测，`openvla_utils` /
# `robot_utils` 是 VEL（openvla/prismatic 那套）的东西，不是本项目的，所以没跟着
# 2026-09-23 那次搬家一起搬过来。采集这条主路一行都不碰它们。
#
# 路径从 `atom_robot.VEL_REPO` 来，可以用环境变量 VEL_REPO 覆盖。
from atom_robot import VEL_REPO  # noqa: E402

if str(VEL_REPO) not in sys.path:
    sys.path.insert(0, str(VEL_REPO))
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_from_server,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Actions executed open-loop before requerying (= NUM_ACTIONS_CHUNK)

    use_vla_server: bool = True                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = "127.0.0.1"   # Remote VLA server host
    vla_server_port: int = 8777                      # Remote VLA server port

    #################################################################################################################
    # AR5 environment parameters
    #################################################################################################################
    num_rollouts_planned: int = 20                   # Number of test rollouts
    max_steps: int = 400                             # Max number of steps per rollout
    control_hz: float = 20.0                         # Outer control-loop rate

    mock: bool = False                               # Run against simulated hardware (no robot required)
    arm_ip: str = "192.168.0.160"                    # ROKAE controller address
    hand_port: str = "/dev/ttyUSB0"                  # Inspire hand RS485 serial port

    max_step_translation: float = 0.02               # Per-step translation limit (m)
    max_step_rotation: float = 0.10                  # Per-step rotation limit (rad)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.use_vla_server, (
        "Must use VLA server (server-client interface) to query model and get actions! "
        "Please set --use_vla_server=True"
    )
    assert cfg.num_open_loop_steps > 0, "num_open_loop_steps must be positive"


def build_env_config(cfg: GenerateConfig) -> AR5Config:
    ar5 = AR5Config(mock=cfg.mock)
    ar5.arm.ip = cfg.arm_ip
    ar5.arm.control_hz = cfg.control_hz
    ar5.arm.max_step_translation = cfg.max_step_translation
    ar5.arm.max_step_rotation = cfg.max_step_rotation
    ar5.hand.port = cfg.hand_port
    return ar5


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-AR5-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")
    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_server_endpoint(cfg: GenerateConfig) -> str:
    ip_address = socket.gethostbyname(str(cfg.vla_server_url))
    return f"http://{ip_address}:{cfg.vla_server_port}/act"


def prepare_observation(obs, resize_size):
    """Build the payload the VLA server expects.

    `get_vla_action` reads `full_image` plus every key containing "wrist", and
    `state` for proprio -- so these key names are load-bearing.
    """
    images = obs.observation["images"]
    full_image = resize_image_for_policy(images["cam_high"], resize_size)
    wrist_image = resize_image_for_policy(images["cam_front"], resize_size)

    observation = {
        "full_image": full_image,
        "wrist_image": wrist_image,
        "state": obs.observation["qpos"],
    }
    return observation, full_image, wrist_image


def get_next_task_label(task_label: str) -> str:
    """Prompt for a new instruction, keeping the previous one on empty input."""
    if task_label == "":
        while True:
            user_input = input("Enter the task name: ").strip()
            if user_input:
                return user_input
            print("Task name cannot be empty for the first episode.")

    user_input = input("Enter the task name (or press Enter to keep the current one): ").strip()
    return user_input if user_input else task_label


def run_episode(cfg, env, task_description, server_endpoint, resize_size, log_file=None):
    step_duration = 1.0 / cfg.control_hz

    log_message("Prepare the scene, then press Enter to begin...", log_file)
    input()

    obs = env.reset()
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images, replay_images_wrist = [], []
    total_model_query_time = 0.0
    episode_start_time = time.time()

    try:
        while t < cfg.max_steps:
            step_start_time = time.time()

            obs = env.get_observation(t=t)
            replay_images.append(obs.observation["images"]["cam_high"])
            replay_images_wrist.append(obs.observation["images"]["cam_front"])

            if len(action_queue) == 0:
                observation, _, _ = prepare_observation(obs, resize_size)
                observation["instruction"] = task_description

                query_start = time.time()
                actions = get_action_from_server(observation, server_endpoint)
                total_model_query_time += time.time() - query_start

                if isinstance(actions, str):
                    raise RuntimeError(f"VLA server returned an error: {actions}")
                action_queue.extend(np.asarray(actions)[: cfg.num_open_loop_steps])

            action = action_queue.popleft()
            obs = env.step(action)
            t += 1

            elapsed = time.time() - step_start_time
            if elapsed < step_duration:
                time.sleep(step_duration - elapsed)
            else:
                log_message(f"t={t}: step overran budget by {(elapsed - step_duration) * 1e3:.1f} ms", log_file)

    except KeyboardInterrupt:
        log_message("\nCaught KeyboardInterrupt: terminating episode early.", log_file)
        env.stop()
    except Exception as exc:  # noqa: BLE001
        log_message(f"\nCaught exception: {exc}", log_file)
        env.stop()

    episode_duration = time.time() - episode_start_time

    log_message(
        f"Clamping: {env.stats['delta_clamped']}/{env.stats['steps']} steps hit the per-step limit, "
        f"{env.stats['workspace_clamped']} hit the workspace box.",
        log_file,
    )

    success = input("Success? Enter 'y' or 'n': ").strip().lower() == "y"

    stats = {
        "success": success,
        "total_steps": t,
        "model_query_time": total_model_query_time,
        "episode_duration": episode_duration,
    }
    return stats, replay_images, replay_images_wrist


def save_episode_videos(cfg, replay_images, replay_images_wrist, episode_idx, success, task_description, log_file=None):
    for frames, notes in ((replay_images, None), (replay_images_wrist, "wrist")):
        save_rollout_video(
            frames,
            episode_idx,
            success=success,
            task_description=task_description,
            fps=int(cfg.control_hz),
            notes=notes,
            log_file=log_file,
        )


@draccus.wrap()
def eval_ar5(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    log_file, _, _ = setup_logging(cfg)
    resize_size = get_image_resize_size(cfg)
    server_endpoint = get_server_endpoint(cfg)
    log_message(f"Querying VLA server at {server_endpoint}", log_file)

    task_description = ""
    num_rollouts_completed, total_successes = 0, 0

    with AR5Env(build_env_config(cfg)) as env:
        for _ in tqdm.tqdm(range(cfg.num_rollouts_planned)):
            task_description = get_next_task_label(task_description)
            log_message(f"\nTask: {task_description}", log_file)
            log_message(f"Starting episode {num_rollouts_completed + 1}...", log_file)

            stats, replay_images, replay_images_wrist = run_episode(
                cfg, env, task_description, server_endpoint, resize_size, log_file
            )

            num_rollouts_completed += 1
            total_successes += int(stats["success"])

            save_episode_videos(
                cfg, replay_images, replay_images_wrist, num_rollouts_completed,
                stats["success"], task_description, log_file,
            )

            log_message(f"Success: {stats['success']}", log_file)
            log_message(f"# episodes so far: {num_rollouts_completed}", log_file)
            log_message(
                f"# successes: {total_successes} ({total_successes / num_rollouts_completed * 100:.1f}%)",
                log_file,
            )
            log_message(f"Total model query time: {stats['model_query_time']:.2f} sec", log_file)
            log_message(f"Episode elapsed time: {stats['episode_duration']:.2f} sec", log_file)

    final_success_rate = total_successes / num_rollouts_completed if num_rollouts_completed else 0.0
    log_message("\nFinal results:", log_file)
    log_message(f"Total episodes: {num_rollouts_completed}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    log_file.close()
    return final_success_rate


if __name__ == "__main__":
    eval_ar5()

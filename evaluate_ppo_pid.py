#!/usr/bin/env python3
"""
Fair randomized comparison: PPO + PID vs IK + PID.

For every episode:
    1. Sample ONE object position; keep the target fixed at the PPO training target.
    2. Run PPO + PID on that exact task.
    3. Run IK + PID on the exact same task.

Only the controller changes.

This evaluates both controllers on identical randomized tasks and avoids
giving either controller an easier/different set of object/target positions.
The existing training environment is NOT modified.
"""

import csv
import importlib
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from rl_env import PandaPickPlaceEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "models" / "ppo_panda_pid_final.zip"
VECNORM_PATH = PROJECT_ROOT / "models" / "ppo_panda_pid_vecnormalize.pkl"

OUT_DIR = PROJECT_ROOT / "evaluation_results_fair_randomized"

N_EPISODES = 50
SEED = 42

# Evaluation distribution matches PPO training:
# - Object position is randomized.
# - Target position is FIXED at the training target.
# This makes the comparison a paired evaluation on the same task distribution
# the PPO policy was actually trained on.
OBJECT_X_RANGE = (0.39, 0.51)
OBJECT_Y_RANGE = (-0.09, 0.09)

FIXED_TARGET_XY = np.array([0.45, -0.30], dtype=np.float64)

OBJECT_Z = 0.05
POSITION_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# PPO evaluation environment
# ---------------------------------------------------------------------------

class FairEvalEnv(PandaPickPlaceEnv):
    """
    Evaluation-only wrapper.

    The normal training reset is allowed to initialize internal state first.
    Then we overwrite object/target positions and restore the arm to the
    normal home pose. This prevents reset-time curriculum from contaminating
    the randomized comparison.

    src/rl_env.py is not modified.
    """

    def reset(self, seed=None, options=None, *, object_xy=None, target_xy=None):
        obs, info = super().reset(seed=seed, options=options)

        if object_xy is None or target_xy is None:
            return obs, info

        object_xy = np.asarray(object_xy, dtype=np.float64)
        target_xy = np.asarray(target_xy, dtype=np.float64)

        # Clean, identical starting state for every randomized evaluation.
        self.data.qpos[:7] = self.home_qpos
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = self.home_qpos

        if hasattr(self, "joint_target"):
            self.joint_target = self.home_qpos.copy()

        # Gripper starts open.
        self.gripper_closed = False
        if hasattr(self, "gripper_actuator"):
            self.data.ctrl[self.gripper_actuator] = 255.0

        # Target position.
        self.target_pos[:2] = target_xy

        # Object position.
        if hasattr(self, "_set_object_pose"):
            self._set_object_pose(
                np.array([object_xy[0], object_xy[1], OBJECT_Z])
            )
        else:
            body = self.model.body("object")
            jnt = int(body.jntadr[0])
            adr = int(self.model.jnt_qposadr[jnt])

            self.data.qpos[adr:adr + 3] = [
                object_xy[0],
                object_xy[1],
                OBJECT_Z,
            ]
            self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

        # Reset task bookkeeping.
        reset_values = {
            "current_step": 0,
            "has_touched_once": False,
            "has_grasped_once": False,
            "has_reached_target_once": False,
            "near_bonus": False,
            "close_bonus": False,
            "consec_holding_steps": 0,
            "consec_not_holding_steps": 0,
            "had_sustained_grasp": False,
            "had_sustained_grasp_since_drop": False,
        }

        for name, value in reset_values.items():
            if hasattr(self, name):
                setattr(self, name, value)

        mujoco.mj_forward(self.model, self.data)

        # Reinitialize reward history from the actual randomized state.
        if hasattr(self, "_get_state"):
            state = self._get_state()
            self.prev_ee_to_obj = float(state[-2])
            self.prev_obj_to_target = float(state[-1])
            self.prev_obj_height = float(state[1][2])

        # Rebuild observation AFTER the override.
        obs = self._get_obs()

        return obs, info


def make_ppo_venv():
    base = DummyVecEnv([lambda: FairEvalEnv()])

    env = VecNormalize.load(str(VECNORM_PATH), base)
    env.training = False
    env.norm_reward = False

    return env


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def safe_float(value, default=np.nan):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def get_distance_mm(info):
    for key in (
        "distance_obj_to_target",
        "final_dist",
        "final_distance",
    ):
        if key in info:
            value = safe_float(info[key])
            if np.isfinite(value):
                return value * 1000.0

    return np.nan


def sample_tasks():
    """
    Generate the object task set ONCE.

    The same randomized object position and the same fixed training target
    are reused by both PPO and IK, guaranteeing paired evaluation.
    """
    rng = np.random.default_rng(SEED)
    tasks = []

    for episode in range(1, N_EPISODES + 1):
        object_xy = np.array([
            rng.uniform(*OBJECT_X_RANGE),
            rng.uniform(*OBJECT_Y_RANGE),
        ], dtype=np.float64)

        tasks.append({
            "episode": episode,
            "object_xy": object_xy,
            "target_xy": FIXED_TARGET_XY.copy(),
        })

    return tasks


# ---------------------------------------------------------------------------
# PPO + PID
# ---------------------------------------------------------------------------

def run_ppo_episode(model, venv, task):
    env = venv.venv.envs[0]

    object_xy = task["object_xy"]
    target_xy = task["target_xy"]

    # Reset with the exact task sampled for this episode.
    raw_obs, _ = env.reset(
        seed=SEED + task["episode"],
        object_xy=object_xy,
        target_xy=target_xy,
    )

    # VecNormalize-normalize the post-override observation.
    obs = venv.normalize_obs(raw_obs[None, ...])

    # Verify that the simulator actually received the requested task.
    actual_object = env.data.xpos[env.object_body_id][:2].copy()
    actual_target = env.target_pos[:2].copy()

    object_error = float(np.linalg.norm(actual_object - object_xy))
    target_error = float(np.linalg.norm(actual_target - target_xy))

    if object_error > POSITION_TOLERANCE:
        raise RuntimeError(
            f"PPO object-position mismatch: {object_error:.3e}"
        )

    if target_error > POSITION_TOLERANCE:
        raise RuntimeError(
            f"PPO target-position mismatch: {target_error:.3e}"
        )

    done = False
    steps = 0
    total_reward = 0.0
    info = {}

    t0 = time.perf_counter()

    while not done and steps < env.max_episode_steps + 1:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = venv.step(action)

        done = bool(dones[0])
        info = infos[0]

        steps += 1
        total_reward += float(reward[0])

    elapsed = time.perf_counter() - t0

    return {
        "experiment": "fair_randomized_comparison",
        "controller": "PPO+PID",
        "episode": task["episode"],
        "success": int(bool(info.get("success", False))),
        "reward": total_reward,
        "steps": steps,
        "wall_time_s": elapsed,
        "final_distance_mm": get_distance_mm(info),
        "max_object_height_mm": safe_float(
            info.get("obj_height", np.nan)
        ) * 1000.0,
        "is_holding": int(bool(info.get("is_holding", False))),
        "object_x": object_xy[0],
        "object_y": object_xy[1],
        "target_x": target_xy[0],
        "target_y": target_xy[1],
        "position_verified": True,
    }


# ---------------------------------------------------------------------------
# IK + PID
# ---------------------------------------------------------------------------

def set_classical_task(module, object_xy, target_xy):
    """
    Apply the exact same object/target task to the classical MuJoCo model.

    live_test_pid.py creates a global model/data pair. We modify that fresh
    model/data before calling its existing run_pick_and_place_pid() function.
    """

    # Object is a freejoint body.
    object_body_id = module.model.body("object").id
    body = module.model.body(object_body_id)

    if body.jntadr[0] < 0:
        raise RuntimeError("Object body does not have the expected freejoint.")

    jnt_id = int(body.jntadr[0])
    qadr = int(module.model.jnt_qposadr[jnt_id])

    module.data.qpos[qadr:qadr + 3] = [
        object_xy[0],
        object_xy[1],
        OBJECT_Z,
    ]

    # Identity quaternion.
    module.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

    # Target is a mocap body.
    target_body_id = module.model.body("target_marker").id
    mocap_id = int(module.model.body_mocapid[target_body_id])

    if mocap_id < 0:
        raise RuntimeError(
            "target_marker is not a mocap body in the classical scene."
        )

    module.data.mocap_pos[mocap_id, :2] = target_xy

    # Keep target Z consistent with the experiment.
    module.data.mocap_pos[mocap_id, 2] = 0.02

    module.data.qvel[:] = 0.0

    mujoco.mj_forward(module.model, module.data)

    # Verify exact task assignment.
    actual_object = module.data.xpos[object_body_id][:2].copy()
    actual_target = module.data.mocap_pos[mocap_id][:2].copy()

    object_error = float(np.linalg.norm(actual_object - object_xy))
    target_error = float(np.linalg.norm(actual_target - target_xy))

    if object_error > POSITION_TOLERANCE:
        raise RuntimeError(
            f"IK object-position mismatch: {object_error:.3e}"
        )

    if target_error > POSITION_TOLERANCE:
        raise RuntimeError(
            f"IK target-position mismatch: {target_error:.3e}"
        )


def run_classical_episode(task):
    """
    Fresh import/reload of live_test_pid.py gives a fresh MuJoCo model/data
    for each episode.
    """

    if "live_test_pid" in sys.modules:
        module = importlib.reload(sys.modules["live_test_pid"])
    else:
        module = importlib.import_module("live_test_pid")

    object_xy = task["object_xy"]
    target_xy = task["target_xy"]

    # Apply the SAME task sampled for PPO.
    set_classical_task(module, object_xy, target_xy)

    t0 = time.perf_counter()

    result = module.run_pick_and_place_pid(
        np.array([target_xy[0], target_xy[1], 0.02]),
        viewer=None,
    )

    elapsed = time.perf_counter() - t0

    return {
        "experiment": "fair_randomized_comparison",
        "controller": "IK+PID",
        "episode": task["episode"],
        "success": int(bool(result.get("success", False))),
        "reward": np.nan,
        "steps": np.nan,
        "wall_time_s": elapsed,
        "final_distance_mm": safe_float(
            result.get("final_dist", np.nan)
        ) * 1000.0,
        "max_object_height_mm": np.nan,
        "is_holding": np.nan,
        "object_x": object_xy[0],
        "object_y": object_xy[1],
        "target_x": target_xy[0],
        "target_y": target_xy[1],
        "position_verified": True,
    }


# ---------------------------------------------------------------------------
# Statistics / plots
# ---------------------------------------------------------------------------

def controller_stats(rows):
    success_rate = 100.0 * np.mean(
        [r["success"] for r in rows]
    )

    distances = np.array(
        [r["final_distance_mm"] for r in rows],
        dtype=float,
    )
    distances = distances[np.isfinite(distances)]

    times = np.array(
        [r["wall_time_s"] for r in rows],
        dtype=float,
    )

    return {
        "success_rate": success_rate,
        "mean_distance": (
            float(np.mean(distances))
            if len(distances)
            else np.nan
        ),
        "median_distance": (
            float(np.median(distances))
            if len(distances)
            else np.nan
        ),
        "mean_time": float(np.mean(times)),
    }


def print_summary(ppo_rows, ik_rows):
    ppo = controller_stats(ppo_rows)
    ik = controller_stats(ik_rows)

    print("\n")
    print("=" * 64)
    print("FAIR RANDOMIZED COMPARISON")
    print("PPO + PID vs IK + PID")
    print("=" * 64)

    print("\nSame randomized object + fixed training target for both controllers.")
    print(f"Episodes: {N_EPISODES}")
    print(f"Seed: {SEED}")

    print("\n--- PPO + PID ---")
    print(f"Success rate:       {ppo['success_rate']:.1f}%")
    print(f"Mean final distance:{ppo['mean_distance']:.2f} mm")
    print(f"Median final dist:  {ppo['median_distance']:.2f} mm")
    print(f"Mean episode time:  {ppo['mean_time']:.3f} s")

    print("\n--- IK + PID ---")
    print(f"Success rate:       {ik['success_rate']:.1f}%")
    print(f"Mean final distance:{ik['mean_distance']:.2f} mm")
    print(f"Median final dist:  {ik['median_distance']:.2f} mm")
    print(f"Mean episode time:  {ik['mean_time']:.3f} s")

    print("\n--- Paired task verification ---")
    print("Every episode uses the same object_x/object_y and")
    print("target_x/target_y for PPO and IK.")
    print("=" * 64)


def write_csv(path, ppo_rows, ik_rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = ppo_rows + ik_rows

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plots(ppo_rows, ik_rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ppo = controller_stats(ppo_rows)
    ik = controller_stats(ik_rows)

    labels = ["PPO + PID", "IK + PID"]

    # Success rate.
    plt.figure(figsize=(7, 5))
    plt.bar(
        labels,
        [ppo["success_rate"], ik["success_rate"]],
    )
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 100)
    plt.title("Fair Randomized Tasks — Success Rate")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "fair_randomized_success_rate.png",
        dpi=200,
    )
    plt.close()

    # Final distance.
    plt.figure(figsize=(7, 5))
    plt.bar(
        labels,
        [ppo["mean_distance"], ik["mean_distance"]],
    )
    plt.ylabel("Mean final distance (mm)")
    plt.title("Fair Randomized Tasks — Placement Accuracy")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "fair_randomized_final_distance.png",
        dpi=200,
    )
    plt.close()

    # Time.
    plt.figure(figsize=(7, 5))
    plt.bar(
        labels,
        [ppo["mean_time"], ik["mean_time"]],
    )
    plt.ylabel("Mean episode time (s)")
    plt.title("Fair Randomized Tasks — Episode Time")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "fair_randomized_episode_time.png",
        dpi=200,
    )
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("FAIR RANDOMIZED PPO + PID vs IK + PID EVALUATION")
    print("=" * 64)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PPO model not found: {MODEL_PATH}\n"
            "Run this script from the project containing models/."
        )

    if not VECNORM_PATH.exists():
        raise FileNotFoundError(
            f"VecNormalize file not found: {VECNORM_PATH}"
        )

    tasks = sample_tasks()

    print("\nTask distribution:")
    print(f"  Object X: {OBJECT_X_RANGE}")
    print(f"  Object Y: {OBJECT_Y_RANGE}")
    print(f"  Target X: fixed at {FIXED_TARGET_XY[0]:.2f}")
    print(f"  Target Y: fixed at {FIXED_TARGET_XY[1]:.2f}")

    # ---------------------------------------------------------------
    # PPO
    # ---------------------------------------------------------------

    print("\n" + "-" * 64)
    print("RUNNING PPO + PID")
    print("-" * 64)

    model = PPO.load(str(MODEL_PATH))
    venv = make_ppo_venv()

    ppo_rows = []

    try:
        for task in tasks:
            row = run_ppo_episode(model, venv, task)
            ppo_rows.append(row)

            print(
                f"PPO+PID {task['episode']:02d}/{N_EPISODES}: "
                f"success={row['success']} "
                f"dist={row['final_distance_mm']:.1f}mm "
                f"object=({task['object_xy'][0]:.3f},"
                f"{task['object_xy'][1]:.3f}) "
                f"target=({task['target_xy'][0]:.3f},"
                f"{task['target_xy'][1]:.3f})"
            )
    finally:
        venv.close()

    # ---------------------------------------------------------------
    # IK
    # ---------------------------------------------------------------

    print("\n" + "-" * 64)
    print("RUNNING IK + PID")
    print("-" * 64)

    ik_rows = []

    for task in tasks:
        row = run_classical_episode(task)
        ik_rows.append(row)

        print(
            f"IK+PID  {task['episode']:02d}/{N_EPISODES}: "
            f"success={row['success']} "
            f"dist={row['final_distance_mm']:.1f}mm "
            f"object=({task['object_xy'][0]:.3f},"
            f"{task['object_xy'][1]:.3f}) "
            f"target=({task['target_xy'][0]:.3f},"
            f"{task['target_xy'][1]:.3f})"
        )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    write_csv(
        OUT_DIR / "fair_randomized_comparison.csv",
        ppo_rows,
        ik_rows,
    )

    print_summary(ppo_rows, ik_rows)
    make_plots(ppo_rows, ik_rows)

    print("\nResults saved to:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()

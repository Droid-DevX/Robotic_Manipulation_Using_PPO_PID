import os
import sys
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


# ------------------------------------------------------------
# Import rl_env.py from src/
# ------------------------------------------------------------

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from rl_env import PandaPickPlaceEnv


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

MODEL_PATH = "models/ppo_panda_pid_final.zip"
VECNORM_PATH = "models/ppo_panda_pid_vecnormalize.pkl"

N_TRIALS = 50


# ------------------------------------------------------------
# EVALUATION ENVIRONMENT
# ------------------------------------------------------------

def make_eval_env(seed=None):

    def make_env():

        env = PandaPickPlaceEnv()

        # Seed the actual Gymnasium environment.
        # VecNormalize.reset() does not accept seed.
        if seed is not None:
            env.reset(seed=seed)

        return env

    base_env = DummyVecEnv([make_env])

    env = VecNormalize.load(
        VECNORM_PATH,
        base_env
    )

    # Evaluation mode.
    env.training = False
    env.norm_reward = False

    return env


# ------------------------------------------------------------
# RUN ONE TRIAL
# ------------------------------------------------------------

def run_trial(model, seed):

    env = make_eval_env(seed)

    raw_env = env.venv.envs[0]

    # VecNormalize.reset() WITHOUT seed.
    obs = env.reset()

    # --------------------------------------------------------
    # Starting object state
    # --------------------------------------------------------

    start_obj = raw_env.data.xpos[
        raw_env.object_body_id
    ].copy()

    start_dist = float(
        np.linalg.norm(
            start_obj[:2]
            -
            raw_env.target_pos[:2]
        )
    )

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    done = False

    steps = 0

    total_reward = 0.0

    max_height = -np.inf

    best_holding_dist = np.inf

    ever_holding = False

    closed_steps = 0

    last_info = {}

    t0 = time.time()

    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    while (
        not done
        and
        steps < raw_env.max_episode_steps + 1
    ):

        # Deterministic evaluation.
        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, dones, infos = env.step(
            action
        )

        done = bool(
            dones[0]
        )

        last_info = infos[0]

        steps += 1

        total_reward += float(
            reward[0]
        )

        # ----------------------------------------------------
        # Object height
        # ----------------------------------------------------

        max_height = max(
            max_height,
            last_info.get(
                "obj_height",
                -np.inf
            )
        )

        # ----------------------------------------------------
        # Holding
        # ----------------------------------------------------

        if last_info.get(
            "is_holding",
            False
        ):

            ever_holding = True

            best_holding_dist = min(
                best_holding_dist,
                last_info.get(
                    "distance_obj_to_target",
                    np.inf
                )
            )

        # ----------------------------------------------------
        # Gripper
        # ----------------------------------------------------

        if float(action[0][7]) > 0.20:

            closed_steps += 1

    elapsed = time.time() - t0

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    final_dist = float(
        last_info.get(
            "distance_obj_to_target",
            np.nan
        )
    )

    result = {

        "success": bool(
            last_info.get(
                "success",
                False
            )
        ),

        "start_dist": start_dist,

        "final_dist": final_dist,

        "net_progress": (
            start_dist
            -
            final_dist
        ),

        "best_holding_dist": (
            best_holding_dist
            if np.isfinite(
                best_holding_dist
            )
            else np.nan
        ),

        "max_height": max_height,

        "ever_holding": ever_holding,

        "closed_frac": (
            closed_steps
            /
            max(steps, 1)
        ),

        "steps": steps,

        "reward": total_reward,

        "time": elapsed,
    }

    env.close()

    return result


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(VECNORM_PATH):

        raise FileNotFoundError(
            f"VecNormalize file not found: "
            f"{VECNORM_PATH}"
        )

    # --------------------------------------------------------
    # Load PPO
    # --------------------------------------------------------

    model = PPO.load(
        MODEL_PATH,
        device="auto"
    )

    # --------------------------------------------------------
    # Run 20 trials
    # --------------------------------------------------------

    results = [
        run_trial(
            model,
            seed=i
        )
        for i in range(N_TRIALS)
    ]

    # --------------------------------------------------------
    # Per-trial results
    # --------------------------------------------------------

    print(
        "\n================ PPO PICK-PLACE EVALUATION ================"
    )

    for i, r in enumerate(results):

        best = (
            r["best_holding_dist"] * 1000
            if np.isfinite(
                r["best_holding_dist"]
            )
            else float("nan")
        )

        print(
            f"Trial {i:02d}: "
            f"success={r['success']} | "
            f"start={r['start_dist'] * 1000:.1f}mm | "
            f"final={r['final_dist'] * 1000:.1f}mm | "
            f"progress={r['net_progress'] * 1000:+.1f}mm | "
            f"best_hold={best:.1f}mm | "
            f"max_height={r['max_height'] * 1000:.1f}mm | "
            f"holding={r['ever_holding']} | "
            f"steps={r['steps']}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    success_rate = (
        100.0
        *
        np.mean([
            r["success"]
            for r in results
        ])
    )

    holding_rate = (
        100.0
        *
        np.mean([
            r["ever_holding"]
            for r in results
        ])
    )

    mean_progress = (
        1000.0
        *
        np.mean([
            r["net_progress"]
            for r in results
        ])
    )

    mean_height = (
        1000.0
        *
        np.mean([
            r["max_height"]
            for r in results
        ])
    )

    finite_best = [
        r["best_holding_dist"]
        for r in results
        if np.isfinite(
            r["best_holding_dist"]
        )
    ]

    print(
        "\n================ SUMMARY ================"
    )

    print(
        f"Success rate:                  "
        f"{success_rate:.1f}%"
    )

    print(
        f"Ever grasped/lifted:           "
        f"{holding_rate:.1f}%"
    )

    print(
        f"Mean net target progress:      "
        f"{mean_progress:+.1f} mm"
    )

    print(
        f"Mean max object height:        "
        f"{mean_height:.1f} mm"
    )

    if finite_best:

        print(
            f"Mean best distance while holding: "
            f"{1000 * np.mean(finite_best):.1f} mm"
        )

    else:

        print(
            "Mean best distance while holding: N/A"
        )
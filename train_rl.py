import os
import sys
sys.path.append("src")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from rl_env import PandaPickPlaceEnv

LOG_DIR = "logs/ppo_panda_pid"
MODEL_DIR = "models"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints_pid")
TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 8

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def make_env():
    return Monitor(PandaPickPlaceEnv())


if __name__ == "__main__":
    # IMPORTANT: this is a fresh PPO run. Do not load the old model or old
    # VecNormalize statistics because the observation/reward/curriculum has
    # changed.
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Evaluation environment starts with the same normalization structure.
    # EvalCallback synchronizes VecNormalize statistics from the training env.
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(20_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        warn=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_panda_pid",
        save_vecnormalize=True,
    )

    callback = CallbackList([eval_callback, checkpoint_callback])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        seed=42,
        device="auto",
    )

    print("Starting FRESH PPO + PID pick-and-place training")
    print(f"Environments: {N_ENVS}")
    print(f"Timesteps:    {TOTAL_TIMESTEPS:,}")
    print("Controller:   PPO -> desired q -> PID -> torque -> MuJoCo")
    print("Object:       randomized x=[0.35,0.55], y=[-0.15,0.15]")
    print("Target:       fixed [0.45,-0.30,0.02]")
    

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=True,
        )
        model.save(os.path.join(MODEL_DIR, "ppo_panda_pid_final"))
        env.save(os.path.join(MODEL_DIR, "ppo_panda_pid_vecnormalize.pkl"))
        print("Training complete.")
        print("Model: models/ppo_panda_pid_final.zip")
        print("VecNormalize: models/ppo_panda_pid_vecnormalize.pkl")
    finally:
        env.close()
        eval_env.close()

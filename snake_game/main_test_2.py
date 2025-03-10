from datetime import datetime
from time import sleep

import gymnasium as gym
import ale_py
import torch.nn as nn
import torch

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from src.tensorboard_video_recorder import TensorboardVideoRecorder

from stable_baselines3.common.atari_wrappers import AtariWrapper 
from stable_baselines3.common.vec_env import VecFrameStack
gym.register_envs(ale_py)

from src.SnakeGameEnv import SnakeGameEnv, Observation_Type
from src.SnakeGame import SnakeGame
from src.utils.mylogger import log, LOG_LEVEL

# Import our exact clone of NatureCNN
from src.custom_cnn import CustomCNN

def snake_game_env_generator(**kwargs) -> SnakeGameEnv:
    return SnakeGameEnv(**kwargs)

MODEL_NAME_ON_DISK = "a2c_resi_L9"
SNAKE_GAME_ID = "SnakeGame"
# Params ==================================
TOTAL_TIMESTEPS = 2500000     # default=1,000,000
TOTAL_EPOCHS = 10
STEPS_PER_EPOCH = 2048
LEARNING_RATE = .00025
IS_SAVING_LOG = False
BOARD_DIMENSION = (10,10)
ARENA_DIMENSION = [8,8]             # None if arena == board
TURN_REWARD = 0         # penalty when turning      # -.015
ALGO = ["ppo", "a2c"][1]

env_kwargs = { 
    "render_mode": "rgb_array",             # do not change
    "obs_type": Observation_Type.IMAGE,     # INPUT: ndarray as multi-input for MlpPolicy. IMAGE: RGB ndarray for CnnPolicy
    "x": BOARD_DIMENSION[0],
    "y": BOARD_DIMENSION[1],
    "arena_size": ARENA_DIMENSION,
    "is_random_spawn": True,
    "is_printing_to_console": False,          # printing on new or on eating
    "turn_reward": TURN_REWARD,
    "fps": None                               # default = None (which is 30)
    }
# =========================================

def show_model_basic(model):
    """Print basic model architecture information"""
    print("\nModel Policy Structure:")
    print(model.policy)
    
    print("\nFeature Extractor (CNN part):")
    print(model.policy.features_extractor)
    
    print("\nTotal trainable parameters:")

def main(is_testing_final=False):
    snakeEnv = SnakeGameEnv(**env_kwargs)
    snake_game_reward_threshold = snakeEnv.game.BOARD_X * snakeEnv.game.BOARD_Y // 2        # Half the board is good
    is_snake_game_deterministic = False
    gym.register(
        SNAKE_GAME_ID, 
        entry_point=snake_game_env_generator, 
        reward_threshold=snake_game_reward_threshold,
        nondeterministic=is_snake_game_deterministic)

    experiment_name = f"{ALGO}_{MODEL_NAME_ON_DISK}"
    experiment_logdir = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{experiment_name}"
    log("main", 
        f"""
        Start training with NatureCNN Clone (should match default). 
        (board:{BOARD_DIMENSION}, arena:{ARENA_DIMENSION})
        (rw_thresh:{snake_game_reward_threshold}) 
        (is_deter:{is_snake_game_deterministic})
        (env_obs_shape:{snakeEnv.observation_space})
        (obs_type:{env_kwargs['obs_type']})
        (algo: {ALGO}),
        (LR={LEARNING_RATE})
        """, log_level=LOG_LEVEL.INFO)

    env = make_vec_env(SNAKE_GAME_ID, n_envs=8, seed=0, env_kwargs=env_kwargs,)

    # Define a trigger function (e.g., record a video every 20,000 steps)
    video_trigger = lambda step: step % 20000 == 0
    # Wrap the environment in a monitor that records videos to Tensorboard
    if IS_SAVING_LOG:
        env = TensorboardVideoRecorder(env=env,
                                    video_trigger=video_trigger,
                                    video_length=2000,
                                    fps=30,
                                    record_video_env_idx=0,
                                    tb_log_dir=experiment_logdir)

    # First run with default CnnPolicy (for comparison in logs)
    default_model = None
    if not is_testing_final:
        log("main", "Creating model with DEFAULT NatureCNN first for comparison", log_level=LOG_LEVEL.INFO)
        if ALGO == "ppo":
            default_model = PPO(
                "CnnPolicy", 
                env, 
                verbose=1,
                tensorboard_log=None  # Don't log the default model
                )
        elif ALGO == "a2c":
            default_model = A2C(
                "CnnPolicy",
                env,
                verbose=1,
                tensorboard_log=None  # Don't log the default model
                )
                
        print("----- DEFAULT MODEL ARCHITECTURE -----")
        show_model_basic(default_model)

    # Define policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        net_arch=[]  # Ensures mlp_extractor remains empty
    )

    # Use a CNN-based policy with the custom features extractor
    if ALGO == "ppo":
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=experiment_logdir if (IS_SAVING_LOG and not is_testing_final) else None,
            gamma=.9,
            n_epochs=TOTAL_EPOCHS,
            n_steps=STEPS_PER_EPOCH,
            learning_rate=LEARNING_RATE,
            policy_kwargs=policy_kwargs
        )
    elif ALGO == "a2c":
        model = A2C(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=experiment_logdir if IS_SAVING_LOG else None,
            gamma=.9,
            policy_kwargs=policy_kwargs
        )

    print(model.policy)
        
    if is_testing_final:            # run to get full trajectory
        model = PPO.load(f"bin/{MODEL_NAME_ON_DISK}") if ALGO == "ppo" else A2C.load(f"bin/{MODEL_NAME_ON_DISK}")
        scores = []

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            env.render("human")

            for info in infos:
                if info["is_terminal"]:
                    scores.append(info["score"])
                    log("main_test", f"Total runs: {len(scores)}\tMean score: {(sum(scores) / len(scores)):4f}", log_level=LOG_LEVEL.INFO)

            sleep(0.1)
            
    else:                           # train and save model to disk
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(f"bin/{MODEL_NAME_ON_DISK}")

if __name__ == '__main__':
    main(is_testing_final=False)
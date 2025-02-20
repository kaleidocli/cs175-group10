from datetime import datetime
from time import sleep

import gymnasium as gym
import ale_py

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from src.tensorboard_video_recorder import TensorboardVideoRecorder

from stable_baselines3.common.atari_wrappers import AtariWrapper 
from stable_baselines3.common.vec_env import VecFrameStack
gym.register_envs(ale_py)

from src.SnakeGameEnv import SnakeGameEnv, Observation_Type
from src.SnakeGame import SnakeGame
from src.utils.mylogger import log, LOG_LEVEL

def snake_game_env_generator(**kwargs) -> SnakeGameEnv:
    return SnakeGameEnv(**kwargs)

MODEL_NAME_ON_DISK = "ppo_snake_pixels"
SNAKE_GAME_ID = "SnakeGame"
# Params ==================================
TOTAL_TIMESTEPS = 10000000     # default=1000000
TOTAL_EPOCHS = 10
STEPS_PER_EPOCH = 2048
IS_SAVING_LOG = False
BOARD_DIMENSION = (10,10)
TURN_REWARD = 0         # penalty when turning      # -.015
ALGO = ["ppo", "a2c"][1]

env_kwargs = { 
    "render_mode": "rgb_array",             # do not change
    "obs_type": Observation_Type.IMAGE,     # INPUT: ndarray as multi-input for MlpPolicy. IMAGE: RGB ndarray for CnnPolicy
    "x": BOARD_DIMENSION[0],
    "y": BOARD_DIMENSION[1],
    "is_random_spawn": True,
    "is_printing_to_console": False,          # printing on new or on eating
    "turn_reward": TURN_REWARD,
    "fps": None                               # default = None (which is 30)
    }
# =========================================

def main(is_testing_final=False):
    snakeEnv = SnakeGameEnv(**env_kwargs)
    snake_game_reward_threshold = snakeEnv.game.BOARD_X * snakeEnv.game.BOARD_Y // 2        # Half the board is good
    is_snake_game_deterministic = False
    gym.register(
        SNAKE_GAME_ID, 
        entry_point=snake_game_env_generator, 
        reward_threshold=snake_game_reward_threshold,
        nondeterministic=is_snake_game_deterministic)

    experiment_name = "a2c_cnn_rspwn_g8_" + SNAKE_GAME_ID
    experiment_logdir = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{experiment_name}"
    log("main", 
        f"""
        Start training. 
        (rw_thresh:{snake_game_reward_threshold}) 
        (is_deter:{is_snake_game_deterministic})
        (env_obs_shape:{snakeEnv.observation_space})
        (obs_type:{env_kwargs['obs_type']})
        """, log_level=LOG_LEVEL.INFO)

    # Instantiate multiple parallel copies of the Breakout environment for faster data collection
    # A Monitor wrapper is included in make_vec_env
    # env = make_vec_env(snake_game_id, n_envs=8, seed=0, wrapper_class=AtariWrapper)
    # env = VecFrameStack(env, n_stack=4)
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

    # Use a CNN-based policy since observations are images.
    if ALGO == "ppo":
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=experiment_logdir if IS_SAVING_LOG else None,
            gamma=.9,    # default=.99
            n_epochs=TOTAL_EPOCHS,
            n_steps=STEPS_PER_EPOCH
            )
    elif ALGO == "a2c":
        model = A2C(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=experiment_logdir if IS_SAVING_LOG else None,
            # device='cpu',
            gamma=.9
        )
    if is_testing_final:            # run to get full trajectory
        model = A2C.load(MODEL_NAME_ON_DISK)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render("human")
            sleep(0.1)
            
    else:                           # train and save model to disk
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        # model.learn(total_timesteps=TOTAL_EPOCHS * STEPS_PER_EPOCH)
        model.save(MODEL_NAME_ON_DISK)

if __name__ == '__main__':
    main(is_testing_final=False)
    
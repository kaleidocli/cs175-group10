from datetime import datetime

import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
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

def main():
    TOTAL_TIMESTEPS = 1000000     # default=1000000
    TOTAL_EPOCHS = 16000
    STEPS_PER_EPOCH = 2048
    IS_SAVING_LOG = True
    BOARD_DIMENSION = (10,10)

    env_kwargs = { 
        "render_mode": "rgb_array",
        "obs_type": Observation_Type.IMAGE,
        "x": BOARD_DIMENSION[0],
        "y": BOARD_DIMENSION[1]
        }
    snakeEnv = SnakeGameEnv(**env_kwargs)
    snake_game_id = "SnakeGame"
    snake_game_reward_threshold = snakeEnv.game.BOARD_X * snakeEnv.game.BOARD_Y // 2        # Half the board is good
    is_snake_game_deterministic = False
    gym.register(
        snake_game_id, 
        entry_point=snake_game_env_generator, 
        reward_threshold=snake_game_reward_threshold,
        nondeterministic=is_snake_game_deterministic)

    experiment_name = "ppo_cnn_cardinal_" + snake_game_id
    experiment_logdir = f"logs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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
    env = make_vec_env(snake_game_id, n_envs=8, seed=0, env_kwargs=env_kwargs)

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
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=experiment_logdir if IS_SAVING_LOG else None,
        gamma=.99,    # default=.99
        n_epochs=TOTAL_EPOCHS,
        n_steps=STEPS_PER_EPOCH
        )
    model.learn(total_timesteps=TOTAL_EPOCHS * STEPS_PER_EPOCH)
    model.save("ppo_snake_pixels")


if __name__ == '__main__':
    main()
    # game = SnakeGame(b_X=50, b_Y=50)
    # game._run()
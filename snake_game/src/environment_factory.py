import gymnasium as gym
from .SnakeGameEnv import SnakeGameEnv

def create_env_factory(env_id, seed, index, run_name, render_mode=None):
    def make_env():
        # Start by initializing the environment itself
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Record videos of the agent
        if index == 0 and render_mode != None:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        # Set the seed in the environment
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # Return the environment
        return env

    return make_env
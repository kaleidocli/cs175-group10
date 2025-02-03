import gymnasium as gym
import numpy as np
from SnakeGame import SnakeGame, Direction

class SnakeGameEnv(gym.Env):
    def __init__(self):
        super(SnakeGameEnv, self).__init__()
        self.game = SnakeGame()
        self.prev_score = 0

        self.observation_shape = (self.game.BOARD_X * self.game.BOARD_X,)
        self.observation_space = gym.spaces.Box(
            low = np.zeros(self.observation_shape), 
            high= np.ones(self.observation_shape), 
            dtype=np.integer
            )
        self.action_space = gym.spaces.Discrete(self.game.MAX_ACTION_COUNT)

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.game = SnakeGame()
        info = dict()
        self.prev_score = 0
        return tuple([self.game.get_observation(), info])
    
    def step(self, action: Direction):
        self.game.step(action)
        reward = self.game.get_score() - self.prev_score
        self.prev_score = self.game.get_score()
        info = dict()

        return tuple([self.game.get_observation(), reward, self.game.is_terminated(), self.game.is_truncated(), info])
    
    def render(self, mode = None):
        if mode == "human":
            return None
        elif mode == "rbg_array":
            return self.game.get_board()
        else:
            return None
        
    def close(self):
        self.game.close()

if __name__ == "__main__":
    env = SnakeGameEnv()
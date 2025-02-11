import gymnasium as gym
import numpy as np
from .SnakeGame import SnakeGame, Direction, LR_Move, Cardinal_Move
from .utils.mylogger import log, LOG_LEVEL

class SnakeGameEnv(gym.Env):
    def __init__(self, **kwargs):
        self._FOOD_REWARD = 1
        self._TURN_REWARD = -.025
        self._TERMINAL_REWARD = -1

        super(SnakeGameEnv, self).__init__()
        self.game = SnakeGame()
        self.prev_score = 0

        self.observation_shape = (self.game.BOARD_X, self.game.BOARD_Y)
        self.observation_space = gym.spaces.Box(
            low = np.zeros(self.observation_shape), 
            high= np.ones(self.observation_shape), 
            dtype=np.int64
            )
        self.action_space = gym.spaces.Discrete(self.game.MAX_ACTION_COUNT)

        self.render_mode = kwargs.get('render_mode', None)
        self.metadata = {
            'render_modes': [None, 'rgb_array']
        }

    def reset(self, seed: int | None = None, options = None):
        log("SnakeGameEnv", "Resetting...")
        super().reset(seed=seed)
        self.game = SnakeGame()
        info = dict()
        self.prev_score = 0
        obs = self.game.get_observation()
        return tuple([obs, info])
    
    def step(self, action):
        self.game.step(action)
        info = dict()

        reward = (self.game.get_score() - self.prev_score) * self._FOOD_REWARD
        # if action != Move.NONE.value:
        #     reward += self._TURN_REWARD
        self.prev_score = self.game.get_score()
        if self.game.is_terminated() or self.game.is_truncated():
            reward = self._TERMINAL_REWARD

        log("SnakeGameEnv", f"Stepped.\trw={reward}")
        return tuple([self.game.get_observation(), reward, self.game.is_terminated(), self.game.is_truncated(), info])
    
    def render(self):
        mode = self.render_mode
        if mode == "human":
            return None
        elif mode == "rgb_array":
            log("SGEnv", "Rendering rgb...")
            return self.game.rgb_render()
            # self.game._IS_RENDERING = True
            # return self.game.get_board()
        else:
            return None

    def seed(self, seed):       # Do nothing
        return
        
    def close(self):
        self.game.close()

if __name__ == "__main__":
    env = SnakeGameEnv()
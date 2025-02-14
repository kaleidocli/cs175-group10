import gymnasium as gym
import numpy as np
from .SnakeGame import SnakeGame, Direction, LR_Move, Cardinal_Move
from .utils.mylogger import log, LOG_LEVEL
from enum import Enum

class Observation_Type(Enum):
    INPUT = 0
    IMAGE = 1

class SnakeGameEnv(gym.Env):
    def __init__(self, **kwargs):
        """
        render_mode:    (str)               rgb_array, human\n
        obs_type:       (Observation_Type)  INPUT, IMAGE
        """

        self._FOOD_REWARD = 1
        self._TURN_REWARD = -.025
        self._TERMINAL_REWARD = -1
        self.OBS_TYPE = kwargs.get("obs_type", Observation_Type.INPUT)
        board_x = kwargs.get("x", 10)
        board_y = kwargs.get("y", 10)

        super(SnakeGameEnv, self).__init__()
        self.game = SnakeGame(b_X=board_x, b_Y=board_y)
        self.prev_score = 0

        # for obs type IMAGE, the shape of obs_space is different from the board shape.
        # - obs_space here is specifically for CNN
        # - board is for game logic
        # CNN does not accept obs_space smaller than 40x40 for some reason, and we want
        # to keep the board space small. Hence why we "upscale" the obs_space from board
        # to give it to CNN.
        self.observation_shape = \
            (self.game.BOARD_X, self.game.BOARD_Y) \
            if self.OBS_TYPE == Observation_Type.INPUT \
            else (3, self.game.BOARD_X * self.game.WINDOW_SIZE_MULTIPLIER, self.game.BOARD_Y * self.game.WINDOW_SIZE_MULTIPLIER)
        if self.OBS_TYPE == Observation_Type.INPUT:
            self.observation_space = gym.spaces.Box(
                low = np.zeros(self.observation_shape), 
                high= np.ones(self.observation_shape), 
                dtype=np.int64
                )
        else:
            self.observation_space = gym.spaces.Box(
                low = 0, 
                high= 255, 
                shape=self.observation_shape,
                dtype=np.uint8
                )
        self.action_space = gym.spaces.Discrete(self.game.MAX_ACTION_COUNT)

        self.render_mode = kwargs.get('render_mode', None)
        self.metadata = {
            'render_modes': [None, 'rgb_array']
        }

    def reset(self, seed: int | None = None, options = None):
        log("SnakeGameEnv", "Resetting...")
        t_bX = self.game.BOARD_X
        t_bY = self.game.BOARD_Y
        super().reset(seed=seed)
        self.game = SnakeGame(b_X=t_bX, b_Y=t_bY)
        info = dict()
        self.prev_score = 0
        obs = self.game.get_observation(is_image_type= self.OBS_TYPE == Observation_Type.IMAGE)
        log("SnakeGameEnv", f"Resetted! (obs:{obs.shape})")
        return tuple([obs, info])
    
    def step(self, action):
        log("SnakeGameEnv", "Stepping...")
        self.game.step(action)
        info = dict()

        reward = (self.game.get_score() - self.prev_score) * self._FOOD_REWARD
        # if action != Move.NONE.value:
        #     reward += self._TURN_REWARD
        self.prev_score = self.game.get_score()
        if self.game.is_terminated() or self.game.is_truncated():
            reward = self._TERMINAL_REWARD

        log("SnakeGameEnv", f"Stepped.\trw={reward}")
        return tuple([
            self.game.get_observation(is_image_type= self.OBS_TYPE == Observation_Type.IMAGE), 
            reward, 
            self.game.is_terminated(), 
            self.game.is_truncated(), 
            info])
    
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
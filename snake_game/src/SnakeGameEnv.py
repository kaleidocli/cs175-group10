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
        self._TURN_REWARD = kwargs.get("turn_reward", 0)
        self._TERMINAL_REWARD = -1
        self.OBS_TYPE = kwargs.get("obs_type", Observation_Type.INPUT)
        board_x = kwargs.get("x", 10)
        board_y = kwargs.get("y", 10)
        arena_size = kwargs.get("arena_size", None)
        self._IS_PRINTING_OBS_TO_CONSOLE = kwargs.get("is_printing_to_console", False)
        self._recorded_obss: list[np.ndarray, list[int], list[int]] = []
        self._MAX_RECORDING_COUNT = 3
        self._is_new = True
        self._is_random_spawn = kwargs.get("is_random_spawn", False)
        self.GAME_FPS = kwargs.get("fps", None)

        super(SnakeGameEnv, self).__init__()
        self.game = SnakeGame(
            b_X=board_x, 
            b_Y=board_y, 
            is_random_spawn=self._is_random_spawn, 
            snake_speed=self.GAME_FPS, 
            arena_size=arena_size)
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
            'render_modes': ['rgb_array'],
            'render_fps': self.game.SNAKE_SPEED
        }

    def reset(self, seed: int | None = None, options = None):
        log("SnakeGameEnv", "Resetting...")
        t_bX = self.game.BOARD_X
        t_bY = self.game.BOARD_Y
        super().reset(seed=seed)
        self.game = SnakeGame(
            b_X=t_bX, 
            b_Y=t_bY, 
            is_random_spawn=self._is_random_spawn, 
            snake_speed=self.GAME_FPS,
            arena_size=[self.game.ARENA_X, self.game.ARENA_Y] if self.game.ARENA_X != None else None)
        info = dict()
        self.prev_score = 0
        obs = self.game.get_observation(is_image_type= self.OBS_TYPE == Observation_Type.IMAGE)
        self._is_new = True
        log("SnakeGameEnv", f"Resetted! (obs:{obs.shape})")
        return tuple([obs, info])
    
    def step(self, action):
        log("SnakeGameEnv", "Stepping...")
        prev_direction = self.game.snake_direction
        if self._IS_PRINTING_OBS_TO_CONSOLE and self._is_new:        # print at spawn
            obs = self.game.get_observation(is_image_type= self.OBS_TYPE == Observation_Type.IMAGE)
            print("=====" * 2 + " New " + "=====" * 2)
            print(f"head: {self.game.snake_bpos}\tbody: {self.game.snake_body_bpos}")
            print(obs)
        self.game.step(action)

        reward = (self.game.get_score() - self.prev_score) * self._FOOD_REWARD
        if prev_direction != self.game.snake_direction:     # penalty if turn
            reward += self._TURN_REWARD
        self.prev_score = self.game.get_score()
        if self.game.is_terminated() or self.game.is_truncated():
            reward = self._TERMINAL_REWARD

        log("SnakeGameEnv", f"Stepped.\trw={reward}")
        obs = self.game.get_observation(is_image_type= self.OBS_TYPE == Observation_Type.IMAGE)
        self._recorded_obss.append((obs, self.game.snake_bpos.copy(), self.game.snake_body_bpos.copy()))
        if len(self._recorded_obss) > self._MAX_RECORDING_COUNT:
            del self._recorded_obss[0]
        if self._IS_PRINTING_OBS_TO_CONSOLE and reward > 0:                   # print at eating
            print("=====" * 2 + f" Scored! Last {self._MAX_RECORDING_COUNT} obss " + "=====" * 2)
            print(f"Rew: {reward}")
            for obs, snake_head, snake_body in self._recorded_obss:
                print(f"head: {snake_head}\tbody: {snake_body}")
                print(obs)
            
        self._is_new = False
        info = {
            "score": self.game.score,
            "is_terminal": self.game.is_terminated() or self.game.is_truncated()
        }
        return tuple([
            obs, 
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
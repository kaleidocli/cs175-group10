import gymnasium as gym
from SnakeGame import SnakeGame, Direction

class SnakeGameEnv(gym.Env):
    def __init__(self):
        self.game = SnakeGame()
        self.prev_score = 0

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
    
    def render(self, mode = "human"):
        if mode == "human":
            return None
        elif mode == "rbg_array":
            return self.game.get_board()
        else:
            return None
        
    def close(self):
        pass

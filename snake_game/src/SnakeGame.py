# BSD 3-Clause License

# Copyright (c) 2021, Pavan Ananth Sharma
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# importing libraries
import pygame
import time
import random
 
import numpy as np
from time import sleep
from enum import Enum

from .utils.mylogger import log, LOG_LEVEL

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Cardinal_Move(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class LR_Move(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2

class Element(Enum):
    NONE = 0
    SNAKE = 1
    FOOD = 2
    OBSTACLE = 3

class SnakeGame:
    def __init__(self, b_X = 10, b_Y = 10):
        self._IS_RENDERING = False
        self._IS_CARDINAL = True    # Snake's noving scheme. Cardinal or Left-right. Note: Cardinal is better
        self.SNAKE_SPEED = 30
        self.BOARD_X = b_X           # minimum 4x4
        self.BOARD_Y = b_Y
        self.BOARD_SHAPE = (self.BOARD_X, self.BOARD_Y)
        self.MAX_ACTION_COUNT = 4 if self._IS_CARDINAL else 3
        
        # Window size
        self.WINDOW_SIZE_MULTIPLIER = 10
        self.WINDOW_X = self._to_window_metric(self.BOARD_X)
        self.WINDOW_Y = self._to_window_metric(self.BOARD_Y)
        
        # defining colors
        self.BLACK = pygame.Color(0, 0, 0)
        self.WHITE = pygame.Color(255, 255, 255)
        self.RED = pygame.Color(255, 0, 0)
        self.GREEN = pygame.Color(0, 255, 0)
        self.BLUE = pygame.Color(0, 0, 255)

        self.game_window: pygame.Surface = None
        self._fps_controller = pygame.time.Clock()
        self.score = 0
        self._is_terminated = False
        self._is_truncated = False
        self._is_first_step = True

        self.INIT_SNAKE_LEN = 3      # minimum = 3
        self.snake_bpos, self.snake_body_bpos = self._generate_snake()
        
        self.food_bpos = self._generate_random_loc_on_board()
        self.does_food_exist = True
        self.snake_direction = Direction.RIGHT

    def get_score(self) -> int:
        return self.score
    
    def is_terminated(self) -> bool:
        return self._is_terminated
    
    def is_truncated(self) -> bool:
        return self._is_truncated

    def _show_score(self, choice, color, font, size):
    
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)
        
        # create the display surface object
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        
        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        
        # displaying text
        self.game_window.blit(score_surface, score_rect)
    
    def _game_over(self):
        
        if self._IS_RENDERING:
            self._render_game_over()

        self._is_terminated = True

    def _render_game_over(self):
        # creating font object my_font
        my_font = pygame.font.SysFont('times new roman', 50)
        
        # creating a text surface on which text
        # will be drawn
        game_over_surface = my_font.render(
            'Your Score is : ' + str(self.score), True, self.RED)
        
        # create a rectangular object for the text
        # surface object
        game_over_rect = game_over_surface.get_rect()
        
        # setting position of the text
        game_over_rect.midtop = (self.WINDOW_X/2, self.WINDOW_Y/4)
        
        # blit wil draw the text on screen
        self.game_window.blit(game_over_surface, game_over_rect)
        pygame.display.flip()
        # after 2 seconds we will quit the program
        # time.sleep(.2)
        
    def close(self):
        # deactivating pygame library
        if self._IS_RENDERING:
            try:
                pygame.quit()
            except TypeError:
                return
        
        # quit the program
        # quit()

    def _get_input_direction_from_keyboard_event(self) -> int | None:
        # handling key events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return Cardinal_Move.LEFT.value
                if event.key == pygame.K_RIGHT:
                    return Cardinal_Move.RIGHT.value
                if event.key == pygame.K_UP:
                    return Cardinal_Move.UP.value
                if event.key == pygame.K_DOWN:
                    return Cardinal_Move.DOWN.value
        return None
    
    def _to_window_metric(self, board_len) -> int:
        return board_len * self.WINDOW_SIZE_MULTIPLIER
    
    def _to_board_metric(self, screen_len) -> int:
        return screen_len // self.WINDOW_SIZE_MULTIPLIER

    def _generate_random_loc_on_board(self) -> list[list[int]]:
        return [random.randrange(1, self.BOARD_X), random.randrange(1, self.BOARD_Y)]
    
    def _generate_snake(self):
        """
        Generate a snake. 
        Initial shape will always be vertical.
        Head:
            - Initial X will be on the second quarter from left to right.
            - Initial Y will be on the third quarter from bottom to top.
        """
        snake_bpos = [self.BOARD_X // 4, self.BOARD_Y // 4 * 2]
        snake_body_bpos = []
        for i in range(self.INIT_SNAKE_LEN):
            snake_body_bpos.append([snake_bpos[0], snake_bpos[1]-i])
        return tuple([snake_bpos, snake_body_bpos])

    def get_board(self) -> np.ndarray:
        board: np.ndarray = np.zeros(self.BOARD_SHAPE, dtype=np.int64)
        log("SnakeGame", f"Getting board. board={board.shape} food={self.food_bpos} snake={self.snake_body_bpos}")
        board[self.food_bpos[0], self.food_bpos[1]] = Element.FOOD.value
        for bpos in self.snake_body_bpos:
            try:
                board[bpos[0], bpos[1]] = Element.SNAKE.value
            except IndexError:
                pass
        return board

    def get_observation(self, is_flatten=False) -> np.ndarray:
        """Get observation, with option to flatten it to (n,) shape"""
        return self.get_board().flatten() if is_flatten else self.get_board()
    
    def _get_direction_given_lr_move(self, lr_move: LR_Move) -> Direction:
        """Return the direction of the snake given left-right move. Based on snake's current direction."""
        moving_left = {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP
        }

        moving_right = {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP
        }

        if lr_move is not LR_Move:
            lr_move = LR_Move(lr_move)

        if lr_move == LR_Move.NONE:
            return self.snake_direction
        if lr_move == LR_Move.LEFT:
            return moving_left[self.snake_direction]
        elif lr_move == LR_Move.RIGHT:
            return moving_right[self.snake_direction]
    
    def _get_direction_given_cardinal_move(self, c_move: Cardinal_Move) -> Direction:
        # if c_move != None: 
        #     log("game", f"cardinal_move:{Cardinal_Move(c_move)}\ttranslated_dir:{Direction(c_move)}", log_level=LOG_LEVEL.INFO)
        if c_move not in self.get_legal_cardinal_moves():
            return self.snake_direction     # if no valid move, keep same direction
        c_move = Cardinal_Move(c_move)
        
        if c_move == Cardinal_Move.UP and self.snake_direction != Direction.DOWN.value:
            self.snake_direction = Direction.UP
        if c_move == Cardinal_Move.DOWN and self.snake_direction != Direction.UP.value:
            self.snake_direction = Direction.DOWN
        if c_move == Cardinal_Move.LEFT and self.snake_direction != Direction.RIGHT.value:
            self.snake_direction = Direction.LEFT
        if c_move == Cardinal_Move.RIGHT and self.snake_direction != Direction.LEFT.value:
            self.snake_direction = Direction.RIGHT
        return self.snake_direction     # if no valid move, keep same direction
    
    def get_legal_cardinal_moves(self) -> list[Direction]:
        actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.snake_direction == Direction.UP:
            actions.remove(Direction.DOWN)
        if self.snake_direction == Direction.DOWN:
            actions.remove(Direction.UP)
        if self.snake_direction == Direction.LEFT:
            actions.remove(Direction.RIGHT)
        if self.snake_direction == Direction.RIGHT:
            actions.remove(Direction.LEFT)
        return [action.value for action in actions]
    
    def step(self, action):
        # If two keys pressed simultaneously
        # we don't want snake to move into two
        # directions simultaneously
        if self._IS_CARDINAL:
            result_direction: Direction = self._get_direction_given_cardinal_move(action)
        else:
            result_direction: Direction = self._get_direction_given_lr_move(action)
        log("SnakeGame", f"Stepping with a={action} d={result_direction} s={self.snake_direction}")
        self.snake_direction = result_direction
    
        # Moving the snake
        if self.snake_direction == Direction.UP:
            self.snake_bpos[1] -= 1
        if self.snake_direction == Direction.DOWN:
            self.snake_bpos[1] += 1
        if self.snake_direction == Direction.LEFT:
            self.snake_bpos[0] -= 1
        if self.snake_direction == Direction.RIGHT:
            self.snake_bpos[0] += 1
    
        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        self.snake_body_bpos.insert(0, list(self.snake_bpos))
        if self.snake_bpos[0] == self.food_bpos[0] and self.snake_bpos[1] == self.food_bpos[1]:
            self.score += 1
            self.does_food_exist = False
        else:
            self.snake_body_bpos.pop()
            
        if not self.does_food_exist:
            self.food_bpos = self._generate_random_loc_on_board()
            
        self.does_food_exist = True

        if self._IS_RENDERING:
            self._render()
    
        # Game Over conditions
        if self.snake_bpos[0] < 0 or self.snake_bpos[0] > self.BOARD_X-1:
            self._is_truncated = True
            log("SnakeGame", "DEATH: by hitting wall")
            self._game_over()
        if self.snake_bpos[1] < 0 or self.snake_bpos[1] > self.BOARD_Y-1:
            self._is_truncated = True
            log("SnakeGame", "DEATH: by hitting wall")
            self._game_over()
    
        # Touching the snake body
        for block in self.snake_body_bpos[1:]:
            if self.snake_bpos[0] == block[0] and self.snake_bpos[1] == block[1]:
                log("SnakeGame", "DEATH: by hitting self")
                self._game_over()

    def rgb_render(self) -> np.ndarray:
        """Return a single frame representing the current state of the environment. 
        A frame is a np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.""" 
        wboard: np.ndarray = np.zeros((self.WINDOW_X, self.WINDOW_Y, 3), dtype=np.int8)
        # Color green
        # food_wpos_x = self._to_window_metric(self.food_bpos[0])
        # food_wpos_y = self._to_window_metric(self.food_bpos[1])
        # wboard[food_wpos_x, food_wpos_y, 0] = 0
        # wboard[food_wpos_x, food_wpos_y, 1] = 255
        # wboard[food_wpos_x, food_wpos_y, 2] = 45
        self._draw_on_wboard(wboard, self.food_bpos[0], self.food_bpos[1], 0, 255, 45)
        # Color white 
        for bpos in self.snake_body_bpos:
            try:
                # wpos_x = self._to_window_metric(bpos[0])
                # wpos_y = self._to_window_metric(bpos[1])
                # wboard[wpos_x, wpos_y, 0] = 255
                # wboard[wpos_x, wpos_y, 1] = 255
                # wboard[wpos_x, wpos_y, 2] = 255
                self._draw_on_wboard(wboard, bpos[0], bpos[1], 255, 255, 255)
            except IndexError:
                pass
        return wboard
    
    def _draw_on_wboard(self, wboard, bpos_anchor_x, bpos_anchor_y, r, g, b):
        """Color at anchor, then upscale downward and rightward starting from the anchor."""
        for x in range(self._to_window_metric(bpos_anchor_x), self._to_window_metric(bpos_anchor_x+1)):
            for y in range(self._to_window_metric(bpos_anchor_y), self._to_window_metric(bpos_anchor_y+1)):
                self._color_wpos(wboard, x, y, r, g, b)
        
    def _color_wpos(self, wboard, wpos_x, wpos_y, r, g, b):
        """Color a window position on the window board"""
        wboard[wpos_x, wpos_y, 0] = r
        wboard[wpos_x, wpos_y, 1] = g
        wboard[wpos_x, wpos_y, 2] = b

    def _render(self):
        if self._is_first_step:
            # Initialising pygame
            pygame.init()
            
            # Initialise game window
            pygame.display.set_caption('Snake Game by: Pavan Ananth Sharma')
            self.game_window = pygame.display.set_mode((self.WINDOW_X, self.WINDOW_Y))

            # FPS (frames per second) controller
            self._fps_controller = pygame.time.Clock()

            self._is_first_step = False

        self.game_window.fill(self.BLACK)
        
        for bpos in self.snake_body_bpos:
            wpos = [self._to_window_metric(bpos[0]), self._to_window_metric(bpos[1])]
            pygame.draw.rect(self.game_window, self.GREEN,
                            pygame.Rect(wpos[0], wpos[1], 10, 10))
        pygame.draw.rect(self.game_window, self.WHITE, pygame.Rect(
            self._to_window_metric(self.food_bpos[0]), self._to_window_metric(self.food_bpos[1]), 10, 10))

        # displaying score countinuously
        self._show_score(1, self.WHITE, 'times new roman', 20)
    
        # Refresh game screen
        pygame.display.update()
    
        # Frame Per Second /Refresh Rate
        self._fps_controller.tick(self.SNAKE_SPEED)

    def _run(self):
        self._IS_RENDERING = True
        self.SNAKE_SPEED = 15
        # Initialising pygame
        pygame.init()
        
        # Initialise game window
        pygame.display.set_caption('Snake Game by: Pavan Ananth Sharma')
        self.game_window = pygame.display.set_mode((self.WINDOW_X, self.WINDOW_Y))

        # FPS (frames per second) controller
        fps = pygame.time.Clock()

        # Main Function
        while True:
            # setting default snake direction towards
            t_move = self._get_input_direction_from_keyboard_event()

            self.step(t_move)

            # displaying score countinuously
            self._show_score(1, self.WHITE, 'times new roman', 20)
        
            # Refresh game screen
            pygame.display.update()

            if self._is_terminated or self._is_truncated:
                self.close()
                quit()
        
            # Frame Per Second /Refresh Rate
            fps.tick(self.SNAKE_SPEED)

# if __name__ == "__main__":
#     myGame = SnakeGame()
#     myGame._run()
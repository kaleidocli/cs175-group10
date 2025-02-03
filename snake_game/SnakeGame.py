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

def log(loc, msg):
    print(f"[{loc}]\t\t{msg}")

class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class SnakeGame:
    def __init__(self):
        self.SNAKE_SPEED = 30
        self.BOARD_X = 80
        self.BOARD_Y = 80
        
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
        self.score = 0

        self.snake_bpos = [10, 5]      # defining snake default position
        self.snake_body_bpos = [[10, 5],
                            [9, 5],
                            [8, 5],
                            [7, 5]
                            ]                   # defining first 4 blocks of snake body
        self.food_bpos = self._generate_random_loc_on_board()
        self.does_food_exist = True
        self.snake_direction = Direction.RIGHT

    def show_score(self, choice, color, font, size):
    
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
    
    def game_over(self):
    
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
        time.sleep(2)
        
        # deactivating pygame library
        pygame.quit()
        
        # quit the program
        quit()

    def _get_input_direction_from_keyboard_event(self) -> Direction | None:
        # handling key events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return Direction.UP
                if event.key == pygame.K_DOWN:
                    return Direction.DOWN
                if event.key == pygame.K_LEFT:
                    return Direction.LEFT
                if event.key == pygame.K_RIGHT:
                    return Direction.RIGHT
        return None
    
    def _to_window_metric(self, board_len) -> int:
        return board_len * self.WINDOW_SIZE_MULTIPLIER
    
    def _to_board_metric(self, screen_len) -> int:
        return screen_len // self.WINDOW_SIZE_MULTIPLIER

    def _generate_random_loc_on_board(self) -> list[list[int]]:
        return [random.randrange(1, self.BOARD_X), random.randrange(1, self.BOARD_Y)]
    
    def get_observation(self) -> np.ndarray:
        board: np.ndarray = np.zeros((self.BOARD_X, self.BOARD_Y))
        board[self.food_bpos[0], self.food_bpos[1]] = 2
        for bpos in self.snake_body_bpos:
            board[bpos[0], bpos[1]] = 1
        return board
    
    def get_legal_actions(self) -> list[Direction]:
        actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.snake_direction == Direction.UP:
            actions.remove(Direction.DOWN)
        if self.snake_direction == Direction.DOWN:
            actions.remove(Direction.UP)
        if self.snake_direction == Direction.LEFT:
            actions.remove(Direction.RIGHT)
        if self.snake_direction == Direction.RIGHT:
            actions.remove(Direction.LEFT)
        return actions
    
    def step(self):
        # setting default snake direction towards
        # right
        input_direction = self.snake_direction

        t_change_to = self._get_input_direction_from_keyboard_event()
        input_direction = t_change_to if t_change_to != None else input_direction
    
        # If two keys pressed simultaneously
        # we don't want snake to move into two
        # directions simultaneously
        if input_direction == Direction.UP and self.snake_direction != Direction.DOWN:
            self.snake_direction = Direction.UP
        if input_direction == Direction.DOWN and self.snake_direction != Direction.UP:
            self.snake_direction = Direction.DOWN
        if input_direction == Direction.LEFT and self.snake_direction != Direction.RIGHT:
            self.snake_direction = Direction.LEFT
        if input_direction == Direction.RIGHT and self.snake_direction != Direction.LEFT:
            self.snake_direction = Direction.RIGHT
    
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
        self.game_window.fill(self.BLACK)
        
        log("SnakeGame.run", f"food_bpos: {self.food_bpos}")
        for bpos in self.snake_body_bpos:
            wpos = [self._to_window_metric(bpos[0]), self._to_window_metric(bpos[1])]
            pygame.draw.rect(self.game_window, self.GREEN,
                            pygame.Rect(wpos[0], wpos[1], 10, 10))
        pygame.draw.rect(self.game_window, self.WHITE, pygame.Rect(
            self._to_window_metric(self.food_bpos[0]), self._to_window_metric(self.food_bpos[1]), 10, 10))
    
        # Game Over conditions
        if self.snake_bpos[0] < 0 or self.snake_bpos[0] > self.BOARD_X-1:
            self.game_over()
        if self.snake_bpos[1] < 0 or self.snake_bpos[1] > self.BOARD_Y-1:
            self.game_over()
    
        # Touching the snake body
        for block in self.snake_body_bpos[1:]:
            if self.snake_bpos[0] == block[0] and self.snake_bpos[1] == block[1]:
                self.game_over()

    def run(self):
        # Initialising pygame
        pygame.init()
        
        # Initialise game window
        pygame.display.set_caption('Snake Game by: Pavan Ananth Sharma')
        self.game_window = pygame.display.set_mode((self.WINDOW_X, self.WINDOW_Y))

        # FPS (frames per second) controller
        fps = pygame.time.Clock()
        
        # fruit
        self.food_bpos = self._generate_random_loc_on_board()
        self.does_food_exist = True

        self.snake_direction = Direction.RIGHT

        # Main Function
        while True:
            self.step()

            # displaying score countinuously
            self.show_score(1, self.WHITE, 'times new roman', 20)
        
            # Refresh game screen
            pygame.display.update()
        
            # Frame Per Second /Refresh Rate
            fps.tick(self.SNAKE_SPEED)

myGame = SnakeGame()
myGame.run()
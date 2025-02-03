---
layout: default
title: Proposal
---

# Project: Snake Game Bot

## 2.2 Summary

Snake Game is a one-player game. The player control a snake to move left, right, up, or down, and gaining score by
eating the fruit. The game ends when the head of snake hit the wall or itself.

Our AI will take the information about the snake, the fruit, and the map as the observation space, and use that
to control the snake to score as much as possible.

In this project, we are using:
- [AgileRL](https://docs.agilerl.com/en/latest/api/algorithms/ppo.html#ppo) as a framework for the AI 
- A modified version of [Snake Game](https://github.com/PavanAnanthSharma/Snake-Game-using-Python/tree/main) by PavanAnanthSharma


## 2.3 AL/ML Algorithm
Snake is a game with perfect observation and random event (which is the position of the fruit).
We will use PPO due to the following reason:
- We believe that the state space of Snake is too large for tabular algorithms like Q-Learning and DQN. 
Therefore, we opt for a policy-based method rather than relying on value-based method.

We believe that PPO is sufficient for our topic, but are always open to incorporating new options if 
we find them fitting. 

## 2.4 Evaluation Plan
1. Sanity test:  Score equals to the longest side of the arena.
   - ie. If the arena is 10x20, then the goal score will be 20
2. Baseline: Score equals half of the area of the arena.
   - ie. If the arena is 10x10, then the goal score will be 50 
3. Goal 1: Same as baseline, but with following conditions
   - Obstacles will randomly appear for an amount of time and disappear after a while.
4. Moonshot goal:
   - There will be another snake using the same policy.
   - A satisfying result in this case will be both snakes surviving and have length of a quarter of the arena.
We expect to reach goal 1. As for moonshot goal, the nature of the game is different so we are not sure if our policy
would apply to the moonshot scenario.

## 2.5 Instructor Meeting Date
1:10 PM, Wednesday, January 29th, 2025
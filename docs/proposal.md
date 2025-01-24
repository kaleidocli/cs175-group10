---
layout: default
title: Proposal
---

# Project: Diplomacy Bot

## 2.2 Summary
Diplomacy is a game of 7 players controlling 7 nations against each other until one nation win.
Each nation has a set of units that they can move on a 2D world map, which is split into zones.
Each unit has a set of legal moves - either move to a zone, support another unit, or do nothing.

Our AI will take the information about the nations and the map as the observation space, and use that to control
a nation to win against other nations.
In this project, we are using:
- [Diplomacy game engine](https://github.com/diplomacy/diplomacy/tree/master)
- [Open_spiel](https://github.com/google-deepmind/open_spiel) as a framework for our AI

## 2.3 AL/ML Algorithm
Diplomacy is game with perfect information and no random event.
We will use Q-Learning due to its properties:
- Model-free: Due to our limitation in data, we do not have a model of players behavior.
- Off-policy: This property offers us better exploration.

## 2.4 Evaluation Plan
We will evaluate our AI through win rate.
We define our winning condition to be either the AI winning the game, or the AI last for a certain amount of time.
We expect the win rate to be 1%, and hoping to improve it to 10% over the time.

At first, we will train against 6 players that move at random to get policy A.
After that, we will train against 6 bots that use policy A, to get policy B.
Do the same to get policy C, D, E, F, G.
Then we will have 7 players using each policy playing against each other, and measure their win rate to see
if each policy is actually better than the last one.
We expect that will be the case, and hope that policy G will have a win rate of 10%.

## 2.5 Instructor Meeting Date
1:10 PM, Wednesday, January 29th, 2025
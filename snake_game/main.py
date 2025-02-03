# Author: Michael Pratt
import os

# import imageio
import gymnasium as gym
import numpy as np
import torch
from agilerl.algorithms.ppo import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import create_population, make_vect_envs
from tqdm import trange

from SnakeGameEnv import SnakeGameEnv


def snake_game_env_generator() -> SnakeGameEnv:
    return SnakeGameEnv()
snakeEnv = SnakeGameEnv()
snake_game_id = "SnakeGame"
snake_game_reward_threshold = snakeEnv.game.BOARD_X * snakeEnv.game.BOARD_Y // 2        # Half the board is good
is_snake_game_deterministic = False
gym.register(
    snake_game_id, 
    entry_point=snake_game_env_generator, 
    reward_threshold=snake_game_reward_threshold,
    nondeterministic=is_snake_game_deterministic)


def train_agent_3():
    # Create environment
    num_envs = 1
    env = make_vect_envs('SnakeGame', num_envs=num_envs)
    try:
        state_dim = env.single_observation_space.n,         # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n              # Discrete action space
    except:
        action_dim = env.single_action_space.shape[0]       # Continuous action space

    channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    discrete_actions = True # Discrete action space

    if channels_last:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    agent = PPO(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot, discrete_actions=discrete_actions)   # Create PPO agent

    while True:
        state, info = env.reset()  # Reset environment at start of episode
        scores = np.zeros(num_envs)
        completed_episode_scores = []
        steps = 0
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        for step in range(agent.learn_step):
            if channels_last:
                state = np.moveaxis(state, [-1], [-3])
            # Get next action from agent
            action, log_prob, _, value = agent.get_action(state)
            print(f"Get action: a={action} =================")
            next_state, reward, done, trunc, _ = env.step(action)  # Act in environment

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state

        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
        )
        # Learn according to agent's RL algorithm
        agent.learn(experiences)



# # Initial hyperparameters
# INIT_HP = {
#     "POP_SIZE": 4,  # Population size
#     "DISCRETE_ACTIONS": True,  # Discrete action space
#     "BATCH_SIZE": 128,  # Batch size
#     "LR": 0.001,  # Learning rate
#     "LEARN_STEP": 1024,  # Learning frequency
#     "GAMMA": 0.99,  # Discount factor
#     "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
#     "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
#     "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
#     "ENT_COEF": 0.01,  # Entropy coefficient
#     "VF_COEF": 0.5,  # Value function coefficient
#     "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
#     "TARGET_KL": None,  # Target KL divergence threshold
#     "UPDATE_EPOCHS": 4,  # Number of policy update epochs
#     # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
#     "CHANNELS_LAST": False,  # Use with RGB states
#     "TARGET_SCORE": 200.0,  # Target score that will beat the environment
#     "MAX_STEPS": 150000,  # Maximum number of steps an agent takes in an environment
#     "EVO_STEPS": 10000,  # Evolution frequency
#     "EVAL_STEPS": None,  # Number of evaluation steps per episode
#     "EVAL_LOOP": 3,  # Number of evaluation episodes
#     "TOURN_SIZE": 2,  # Tournament size
#     "ELITISM": True,  # Elitism in tournament selection
# }

# # Mutation parameters
# MUT_P = {
#     # Mutation probabilities
#     "NO_MUT": 0.4,  # No mutation
#     "ARCH_MUT": 0.2,  # Architecture mutation
#     "NEW_LAYER": 0.2,  # New layer mutation
#     "PARAMS_MUT": 0.2,  # Network parameters mutation
#     "ACT_MUT": 0.2,  # Activation layer mutation
#     "RL_HP_MUT": 0.2,  # Learning HP mutation
#     # Learning HPs to choose from
#     "RL_HP_SELECTION": ["lr", "batch_size", "learn_step"],
#     "MUT_SD": 0.1,  # Mutation strength
#     "RAND_SEED": 42,  # Random seed
#     # Define max and min limits for mutating RL hyperparams
#     "MIN_LR": 0.0001,
#     "MAX_LR": 0.01,
#     "MIN_BATCH_SIZE": 8,
#     "MAX_BATCH_SIZE": 1024,
#     "MIN_LEARN_STEP": 256,
#     "MAX_LEARN_STEP": 8192,
# }



# num_envs=8
# env = make_vect_envs(snake_game_id, num_envs=num_envs)  # Create environment
# try:
#     state_dim = env.single_observation_space.n, # Discrete observation space
#     one_hot = True  # Requires one-hot encoding
# except Exception:
#     state_dim = env.single_observation_space.shape  # Continuous observation space
#     one_hot = False  # Does not require one-hot encoding
# try:
#     action_dim = env.single_action_space.n  # Discrete action space
# except Exception:
#     action_dim = env.single_action_space.shape[0]  # Continuous action space

# if INIT_HP["CHANNELS_LAST"]:
#     # Adjust dimensions for PyTorch API (C, H, W), for envs with RGB image states
#     state_dim = (state_dim[2], state_dim[0], state_dim[1])



# # Set-up the device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
# net_config = {"arch": "mlp", "hidden_size": [64, 64]}

# # Define a population
# pop = create_population(
#     algo="PPO",  # Algorithm
#     state_dim=state_dim,  # State dimension
#     action_dim=action_dim,  # Action dimension
#     one_hot=one_hot,  # One-hot encoding
#     net_config=net_config,  # Network configuration
#     INIT_HP=INIT_HP,  # Initial hyperparameter
#     population_size=INIT_HP["POP_SIZE"],  # Population size
#     num_envs=num_envs,
#     device=device,
# )



# tournament = TournamentSelection(
#     INIT_HP["TOURN_SIZE"],
#     INIT_HP["ELITISM"],
#     INIT_HP["POP_SIZE"],
#     INIT_HP["EVAL_LOOP"],
# )



# mutations = Mutations(
#     algo="PPO",
#     no_mutation=MUT_P["NO_MUT"],
#     architecture=MUT_P["ARCH_MUT"],
#     new_layer_prob=MUT_P["NEW_LAYER"],
#     parameters=MUT_P["PARAMS_MUT"],
#     activation=MUT_P["ACT_MUT"],
#     rl_hp=MUT_P["RL_HP_MUT"],
#     rl_hp_selection=MUT_P["RL_HP_SELECTION"],
#     min_lr=MUT_P["MIN_LR"],
#     max_lr=MUT_P["MAX_LR"],
#     min_batch_size=MUT_P["MAX_BATCH_SIZE"],
#     max_batch_size=MUT_P["MAX_BATCH_SIZE"],
#     min_learn_step=MUT_P["MIN_LEARN_STEP"],
#     max_learn_step=MUT_P["MAX_LEARN_STEP"],
#     mutation_sd=MUT_P["MUT_SD"],
#     arch=net_config["arch"],
#     rand_seed=MUT_P["RAND_SEED"],
#     device=device,
# )



# def train_agent():
#     # Define a save path for our trained agent
#     save_path = "PPO_trained_agent.pt"

#     trained_pop, pop_fitnesses = train_on_policy(
#         env=env,
#         env_name=snake_game_id,
#         algo="PPO",
#         pop=pop,
#         INIT_HP=INIT_HP,
#         MUT_P=MUT_P,
#         swap_channels=INIT_HP["CHANNELS_LAST"],
#         max_steps=INIT_HP["MAX_STEPS"],
#         evo_steps=INIT_HP["EVO_STEPS"],
#         eval_steps=INIT_HP["EVAL_STEPS"],
#         eval_loop=INIT_HP["EVAL_LOOP"],
#         tournament=tournament,
#         mutation=mutations,
#         wb=False,  # Boolean flag to record run with Weights & Biases
#         save_elite=True,  # Boolean flag to save the elite agent in the population
#         elite_path=save_path,
#     )

# def train_agent_2(pop):
#     # Define a save path for our trained agent
#     save_path = "PPO_trained_agent.pt"

#     total_steps = 0

#     # TRAINING LOOP
#     print("Training...")
#     pbar = trange(INIT_HP["MAX_STEPS"], unit="step")
#     while np.less([agent.steps[-1] for agent in pop], INIT_HP["MAX_STEPS"]).all():
#         pop_episode_scores = []
#         for agent in pop:  # Loop through population
#             state, info = env.reset()  # Reset environment at start of episode
#             scores = np.zeros(num_envs)
#             completed_episode_scores = []
#             steps = 0

#             for _ in range(-(INIT_HP["EVO_STEPS"] // -agent.learn_step)):

#                 states = []
#                 actions = []
#                 log_probs = []
#                 rewards = []
#                 dones = []
#                 values = []

#                 learn_steps = 0

#                 for idx_step in range(-(agent.learn_step // -num_envs)):
#                     if INIT_HP["CHANNELS_LAST"]:
#                         state = np.moveaxis(state, [-1], [-3])

#                     # Get next action from agent
#                     action, log_prob, _, value = agent.get_action(state)
#                     print(f"Getting action: {type(agent)} ========================")

#                     # Act in environment
#                     next_state, reward, terminated, truncated, info = env.step(action)

#                     total_steps += num_envs
#                     steps += num_envs
#                     learn_steps += num_envs

#                     states.append(state)
#                     actions.append(action)
#                     log_probs.append(log_prob)
#                     rewards.append(reward)
#                     dones.append(terminated)
#                     values.append(value)

#                     state = next_state
#                     scores += np.array(reward)

#                     for idx, (d, t) in enumerate(zip(terminated, truncated)):
#                         if d or t:
#                             completed_episode_scores.append(scores[idx])
#                             agent.scores.append(scores[idx])
#                             scores[idx] = 0

#                 pbar.update(learn_steps // len(pop))

#                 if INIT_HP["CHANNELS_LAST"]:
#                     next_state = np.moveaxis(next_state, [-1], [-3])

#                 experiences = (
#                     states,
#                     actions,
#                     log_probs,
#                     rewards,
#                     dones,
#                     values,
#                     next_state,
#                 )
#                 # Learn according to agent's RL algorithm
#                 agent.learn(experiences)

#             agent.steps[-1] += steps
#             pop_episode_scores.append(completed_episode_scores)

#         # Evaluate population
#         fitnesses = [
#             agent.test(
#                 env,
#                 swap_channels=INIT_HP["CHANNELS_LAST"],
#                 max_steps=INIT_HP["EVAL_STEPS"],
#                 loop=INIT_HP["EVAL_LOOP"],
#             )
#             for agent in pop
#         ]
#         mean_scores = [
#             (
#                 np.mean(episode_scores)
#                 if len(episode_scores) > 0
#                 else "0 completed episodes"
#             )
#             for episode_scores in pop_episode_scores
#         ]

#         print(f"--- Global steps {total_steps} ---")
#         print(f"Steps {[agent.steps[-1] for agent in pop]}")
#         print(f"Scores: {mean_scores}")
#         print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
#         print(
#             f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
#         )

#         # Tournament selection and population mutation
#         elite, pop = tournament.select(pop)
#         pop = mutations.mutation(pop)

#         # Update step counter
#         for agent in pop:
#             agent.steps.append(agent.steps[-1])

#     # Save the trained algorithm
#     elite.save_checkpoint(save_path)

#     pbar.close()
#     env.close()

if __name__ == "__main__":
    train_agent_3()
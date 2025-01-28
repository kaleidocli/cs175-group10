# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Tic tac toe (noughts and crosses), implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

import numpy as np
import diplomacy as dp
import random
from dipTranslator import DipTranslator
from tabulate import tabulate

dipTrans = DipTranslator()
_NUM_LOCS = len(dipTrans.loc_to_idx)
_NUM_POWS = len(dipTrans.power_to_idx)
_NUM_MOVES = 4
_NUM_STATS = 3

_NUM_PLAYERS = 3  # adjusted
_NUM_ROWS = _NUM_LOCS
_NUM_COLS = _NUM_POWS
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="python_dip",      # adjusted
    long_name="Python Diplomacy", # adjusted
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,      # adjusted REWARDS if want to update on the fly
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_MOVES * _NUM_LOCS,    # adjusted
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,    # adjusted
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=120)    # adjusted


class DipGame(pyspiel.Game):
  """A Python version of the Tic-Tac-Toe game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return DipState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class DipState(pyspiel.State):
  """A python version of the Tic-Tac-Toe state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)

    self.dpGame = dp.Game()
    # Limit player count
    temp_players = random.sample(self.dpGame.powers.items(), _NUM_PLAYERS)
    self.dpPlayers = dict[str, dp.Power]
    self.dpPlayerNames = list(self.dpPlayers.keys())
    self.dpPlayerScores = [0.0] * len(self.dpPlayers)
    for k, v in temp_players:
      self.dpPlayers[k] = v
      self.dpPlayerNames.append(k)
    for power_name, power in self.dpGame.powers.items():
      power: dp.Power
      if power_name in self.dpPlayers.keys():
        continue
      power.clear_centers()
      power.clear_units()

    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

  def _legal_actions(self, player):
    """
    Returns a list of legal actions, sorted in ascending order.
    --- cs175 ---
    An action is an alpha sorted list of orders, each corresponds to a valid unit
    and does not overlaps.
    """
    locs_and_orders = self.dpGame.get_all_possible_orders()
    return self._cartesian_product_of_orders_of_locs(locs_and_orders, self.dpGame.get_orderable_locations(player))

  def _apply_action(self, action):
    """
    Applies the specified action to the state.
    --- cs175 ---
    An action is an alpha sorted list of orders, each corresponds to a valid unit
    and does not overlaps.
    """
    self.dpGame.set_orders(self.dpPlayerNames[self._cur_player], action)
    if self.dpGame.is_game_done:    # Check terminal state
      self._is_terminal = True
      for i in range(len(self.dpPlayers)):
        self.dpPlayerScores[i] = 1.0 if self.dpPlayerNames[i] in self.dpGame.outcome else -1.0
    else:
      self._cur_player = 0 if self._cur_player == len(self.dpPlayers) else self._cur_player + 1
    self.dpGame.process()
      
  def _action_to_string(self, player, action) -> str:
    """Action -> string."""
    return str(action)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self.dpPlayerScores

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "DipState.__str__ not implemented"
  
  def _cartesian_product_of_orders_of_locs(self, locs_and_orders: dict, locs, current_orders: list=[], index=0) -> list[list]:
    # Base case: if we've reached the end of the lists, add the sequence
    if index == len(locs_and_orders):
        return [current_orders.sort()]
    
    # Recursive case: iterate through the current list and append results
    result = []
    for order in locs_and_orders[locs[index]]:
        result.extend(self._cartesian_product_of_orders_of_locs(locs_and_orders, locs, current_orders.append(order), index + 1))
    return result


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    shape = (_NUM_STATS, _NUM_POWS, _NUM_LOCS)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state: DipState, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    translated_game_state: list[list[list[int]]] = \
      dipTrans.translate_game_state_to_matrix(state.dpGame.get_state(), _NUM_LOCS, _NUM_POWS, _NUM_STATS)

    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs.fill(0)
    for x in range(_NUM_LOCS):
      for y in range(_NUM_POWS):
        for z in range(_NUM_STATS):
          obs[x, y, z] = translated_game_state[x][y][z]

  def string_from(self, state: DipState, player) -> str:
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _mtx_to_str(None)
  
def _mtx_to_str(mtx: list[list[list[int]]]) -> str:
  return "Not implemented"


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, DipGame)

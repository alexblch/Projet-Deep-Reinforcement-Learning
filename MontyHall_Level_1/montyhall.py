import numpy as np
from typing import Tuple
import random

class MontyHallEnv:
    def __init__(self):
        self.num_doors = 3
        self.reset()

    @staticmethod
    def from_random_state() -> 'MontyHallEnv':
        env = MontyHallEnv()
        env.winning_door = np.random.randint(env.num_doors)
        env.selected_door = np.random.randint(env.num_doors)
        env.opened_door = env._open_door()
        return env

    def __call__(self) -> 'MontyHallEnv':
        return self

    def num_states(self) -> int:
        return self.num_doors * (self.num_doors + 1)  # state: (selected_door, opened_door) + initial state (None, None)

    def num_actions(self) -> int:
        return 2  # Actions: Stick with initial choice, or switch

    def num_rewards(self) -> int:
        return 2  # Rewards: 1 (win), 0 (lose)

    def reward(self, i: int) -> float:
        if i == 0:
            return 0.0  # Lose
        elif i == 1:
            return 1.0  # Win
        else:
            return 0.0  # Default reward for non-terminal states

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        original_state = self.state
        self.state = self.index_to_state(s)

        if self.is_game_over():
            self.state = original_state
            return 0.0  # No transition possible if the game is over

        next_state, reward, done = self.step(a)
        probability = 1.0 if next_state == s_p and reward == self.reward(r_index) else 0.0

        self.state = original_state
        return probability

    def state_id(self) -> int:
        return self.state_to_index(self.state)

    def reset(self):
        self.winning_door = np.random.randint(self.num_doors)
        self.selected_door = None
        self.opened_door = None
        self.state = (self.selected_door, self.opened_door)
        return self.state_to_index(self.state)

    def is_game_over(self) -> bool:
        return self.selected_door is not None and self.opened_door is not None and self.state[0] is not None

    def available_actions(self) -> np.ndarray:
        return np.array([0, 1])  # 0: Stick, 1: Switch

    def step(self, action: int):
        if self.selected_door is None:
            self.selected_door = action
            self.opened_door = self._open_door()
            reward = 0
            done = False
        else:
            if action == 1:
                self.selected_door = [d for d in range(self.num_doors) if d != self.selected_door and d != self.opened_door][0]
            reward = 1 if self.selected_door == self.winning_door else 0
            done = True

        self.state = (self.selected_door, self.opened_door)
        return self.state_to_index(self.state), reward, done

    def step_with_state_return(self, action: int) -> Tuple[int, int]:
        next_state, _, _ = self.step(action)
        return self.index_to_state(next_state)

    def score(self) -> float:
        return 1.0 if self.selected_door == self.winning_door else 0.0

    def display(self):
        print(f"Winning Door: {self.winning_door}, Selected Door: {self.selected_door}, Opened Door: {self.opened_door}")

    def _open_door(self):
        doors = [d for d in range(self.num_doors) if d != self.selected_door and d != self.winning_door]
        return np.random.choice(doors)

    def state_to_index(self, state):
        selected_door, opened_door = state
        if selected_door is None or opened_door is None:
            return 0
        return selected_door * self.num_doors + opened_door + 1

    def index_to_state(self, index):
        if index == 0:
            return (None, None)
        index -= 1
        selected_door = index // self.num_doors
        opened_door = index % self.num_doors
        return (selected_door, opened_door)


class MontyHall:
    def __init__(self):
        self.num_doors = 3
        self.reset()

    @staticmethod
    def from_random_state() -> 'MontyHallEnv':
        env = MontyHallEnv()
        env.winning_door = np.random.randint(env.num_doors)
        env.selected_door = np.random.randint(env.num_doors)
        env.opened_door = env._open_door()
        return env

    def __call__(self) -> 'MontyHallEnv':
        return self

    def num_states(self) -> int:
        return self.num_doors * (self.num_doors + 1)  # state: (selected_door, opened_door) + initial state (None, None)

    def num_actions(self) -> int:
        return 2  # Actions: Stick with initial choice, or switch

    def num_rewards(self) -> int:
        return 2  # Rewards: 1 (win), 0 (lose)

    def reward(self, i: int) -> float:
        if i == 0:
            return 0.0  # Lose
        elif i == 1:
            return 1.0  # Win
        else:
            return 0.0  # Default reward for non-terminal states

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        original_state = self.state
        self.state = self.index_to_state(s)

        if self.is_game_over():
            self.state = original_state
            return 0.0  # No transition possible if the game is over

        next_state, reward, done = self.step(a)
        probability = 1.0 if next_state == s_p and reward == self.reward(r_index) else 0.0

        self.state = original_state
        return probability

    def state_id(self) -> int:
        return self.state_to_index(self.state)

    def reset(self):
        self.winning_door = np.random.randint(self.num_doors)
        self.selected_door = None
        self.opened_door = None
        self.state = (self.selected_door, self.opened_door)
        return self.state_to_index(self.state)

    def is_game_over(self) -> bool:
        return self.selected_door is not None and self.opened_door is not None and self.state[0] is not None

    def available_actions(self) -> np.ndarray:
        return np.array([0, 1])  # 0: Stick, 1: Switch

    def step(self, action: int):
        if self.selected_door is None:
            self.selected_door = action
            self.opened_door = self._open_door()
            reward = 0
            done = False
        else:
            if action == 1:
                self.selected_door = [d for d in range(self.num_doors) if d != self.selected_door and d != self.opened_door][0]
            reward = 1 if self.selected_door == self.winning_door else 0
            done = True

        self.state = (self.selected_door, self.opened_door)
        return self.state_to_index(self.state), reward, done

    def step_with_state_return(self, action: int) -> Tuple[int, int]:
        next_state, _, _ = self.step(action)
        return self.index_to_state(next_state)

    def score(self) -> float:
        return 1.0 if self.selected_door == self.winning_door else 0.0

    def display(self):
        print(f"Winning Door: {self.winning_door}, Selected Door: {self.selected_door}, Opened Door: {self.opened_door}")

    def _open_door(self):
        doors = [d for d in range(self.num_doors) if d != self.selected_door and d != self.winning_door]
        return np.random.choice(doors)

    def state_to_index(self, state):
        selected_door, opened_door = state
        if selected_door is None or opened_door is None:
            return 0
        return selected_door * self.num_doors + opened_door + 1

    def index_to_state(self, index):
        if index == 0:
            return (None, None)
        index -= 1
        selected_door = index // self.num_doors
        opened_door = index % self.num_doors
        return (selected_door, opened_door)

import numpy as np
from itertools import combinations
from typing import Tuple


class MontyHallEnvLevel2:
    def __init__(self):
        self.num_doors = 5
        self.reset()

    @staticmethod
    def from_random_state() -> 'MontyHallEnvLevel2':
        env = MontyHallEnvLevel2()
        env.winning_door = np.random.randint(env.num_doors)
        env.selected_door = np.random.randint(env.num_doors)
        env.opened_doors = env._open_doors()
        return env

    def __call__(self) -> 'MontyHallEnvLevel2':
        return self

    def num_states(self) -> int:
        from math import comb
        # Calculate the number of states:
        # (number of doors + 1) for selected door (including None),
        # multiplied by combinations of 3 opened doors from the remaining doors.
        return (self.num_doors + 1) * comb(self.num_doors - 1, 3) + 1

    def num_actions(self) -> int:
        return 2  # Actions: Stick with initial choice, or switch

    def num_rewards(self) -> int:
        return 2  # Rewards: 1 (win), 0 (lose)

    def reward(self, i: int) -> float:
        return 1.0 if i == 1 else 0.0

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
        self.opened_doors = []
        self.state = (self.selected_door, tuple(self.opened_doors))
        return self.state_to_index(self.state)

    def is_game_over(self) -> bool:
        return self.selected_door is not None and len(self.opened_doors) == 3

    def available_actions(self) -> np.ndarray:
        return np.array([0, 1])  # 0: Stick, 1: Switch

    def step(self, action: int):
        if self.selected_door is None:
            self.selected_door = action
            self.opened_doors = self._open_doors()
            reward = 0
            done = False
        else:
            if action == 1:
                remaining_doors = [d for d in range(self.num_doors) if d != self.selected_door and d not in self.opened_doors]
                self.selected_door = remaining_doors[0]
            reward = 1 if self.selected_door == self.winning_door else 0
            done = True

        self.state = (self.selected_door, tuple(self.opened_doors))
        return self.state_to_index(self.state), reward, done

    def step_with_state_return(self, action: int) -> Tuple[int, int]:
        next_state, _, _ = self.step(action)
        return self.index_to_state(next_state)

    def score(self) -> float:
        return 1.0 if self.selected_door == self.winning_door else 0.0

    def display(self):
        print(f"Winning Door: {self.winning_door}, Selected Door: {self.selected_door}, Opened Doors: {self.opened_doors}")

    def _open_doors(self):
        doors = [d for d in range(self.num_doors) if d != self.selected_door and d != self.winning_door]
        return np.random.choice(doors, 3, replace=False).tolist()

    def state_to_index(self, state):
        selected_door, opened_doors = state
        if selected_door is None:
            return 0
        opened_doors = sorted(opened_doors)
        opened_doors_index = sum([door * (self.num_doors ** i) for i, door in enumerate(opened_doors)])
        index = (selected_door + 1) * (self.num_doors ** 3) + opened_doors_index
        return index

    def index_to_state(self, index):
        if index == 0:
            return (None, tuple())
        index -= 1
        selected_door = index // (self.num_doors ** 3)
        opened_doors_index = index % (self.num_doors ** 3)
        opened_doors = []
        for _ in range(3):
            opened_doors.append(opened_doors_index % self.num_doors)
            opened_doors_index //= self.num_doors
        return (selected_door, tuple(sorted(opened_doors)))

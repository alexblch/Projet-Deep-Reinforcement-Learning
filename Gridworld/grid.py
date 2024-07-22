import numpy as np
from typing import List, Tuple
from tqdm import tqdm



class GridWorld:
    def __init__(self):
        self.agent_pos_col = 0
        self.agent_pos_row = 0
        
    @staticmethod
    def from_random_state() -> 'GridWorld':
        env = GridWorld()
        env.agent_pos_col = np.random.randint(5)
        env.agent_pos_row = np.random.randint(5)
        return env
      
    def is_game_over(self) -> bool:
        if self.agent_pos_row <= 4 and self.agent_pos_row >= 0 and self.agent_pos_col <= 4 and self.agent_pos_col >= 0:
            if (self.agent_pos_row != 4 or self.agent_pos_col != 4) and (self.agent_pos_row != 0 or self.agent_pos_col != 4):
                return False
        return True
  
    def state_id(self) -> Tuple[int, int]:
        return self.agent_pos_col, self.agent_pos_row

    def step(self, action: int):
        assert(not self.is_game_over())
        if action == 0:  
            self.agent_pos_col -= 1
        elif action == 1:
            self.agent_pos_col += 1
        elif action == 2:  
            self.agent_pos_row -= 1
        elif action == 3:
            self.agent_pos_row += 1

    def score(self) -> float:
        if self.agent_pos_row == 0 and self.agent_pos_col == 4:
            return -3.0
        if self.agent_pos_row == 4 and self.agent_pos_col == 4:
            return 1.0
        if self.agent_pos_row > 4 or self.agent_pos_row < 0 or self.agent_pos_col > 4 or self.agent_pos_col < 0:
            return -1.0
        return 0.0

    def display(self):
        for num_row in range(5):
            for num_col in range(5):
                print('X' if self.agent_pos_col == num_col and self.agent_pos_row == num_row else '_', end=' ')
            print()

    def reset(self):
        self.agent_pos_col = 0
        self.agent_pos_row = 0
        
    def available_actions(self):
        actions = []
        if self.agent_pos_col > 0:  # Peut aller à gauche
            actions.append(0)
        if self.agent_pos_col < 4:  # Peut aller à droite
            actions.append(1)
        if self.agent_pos_row > 0:  # Peut aller en haut
            actions.append(2)
        if self.agent_pos_row < 4:  # Peut aller en bas
            actions.append(3)
        return actions
    
    def length(self) -> Tuple[int, int]:
        return (5, 5)
    
    @staticmethod
    def size() -> Tuple[int, int]:
        return (5, 5)

class GridWorldC:
    def __init__(self):
        self.agent_pos_col = 0
        self.agent_pos_row = 0

    @staticmethod
    def from_random_state() -> 'GridWorldC':
        env = GridWorldC()
        env.agent_pos_col = np.random.randint(5)
        env.agent_pos_row = np.random.randint(5)
        return env

    def __call__(self) -> 'GridWorldC':
        return self

    def num_states(self) -> int:
        return 25  # 5x5 grid

    def num_actions(self) -> int:
        return 4  # Left, Right, Up, Down

    def num_rewards(self) -> int:
        return 3  # There are 3 unique rewards: -3, 1, -1

    def reward(self, i: int) -> float:
        if i == 0:
            return -3.0
        elif i == 1:
            return 1.0
        elif i == 2:
            return -1.0
        else:
            return 0.0  # Default reward for non-terminal states

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        original_pos_col = self.agent_pos_col
        original_pos_row = self.agent_pos_row

        self.agent_pos_col, self.agent_pos_row = divmod(s, 5)
        if self.is_game_over():
            return 0.0  # No transition possible if the game is over

        next_state = self.step_with_state_return(a)
        next_state_id = next_state[0] * 5 + next_state[1]

        probability = 1.0 if next_state_id == s_p and self.score() == self.reward(r_index) else 0.0

        self.agent_pos_col = original_pos_col
        self.agent_pos_row = original_pos_row

        return probability

    def state_id(self) -> int:
        return self.agent_pos_col * 5 + self.agent_pos_row

    def reset(self):
        self.agent_pos_col = 0
        self.agent_pos_row = 0

    def is_forbidden(self, action: int) -> int:
        if action == 0 and self.agent_pos_col == 0:
            return 1
        if action == 1 and self.agent_pos_col == 4:
            return 1
        if action == 2 and self.agent_pos_row == 0:
            return 1
        if action == 3 and self.agent_pos_row == 4:
            return 1
        return 0

    def is_game_over(self) -> bool:
        return (self.agent_pos_row == 4 and self.agent_pos_col == 4) or (self.agent_pos_row == 0 and self.agent_pos_col == 4)

    def available_actions(self) -> np.ndarray:
        actions = []
        if self.agent_pos_col > 0:  # Can move left
            actions.append(0)
        if self.agent_pos_col < 4:  # Can move right
            actions.append(1)
        if self.agent_pos_row > 0:  # Can move up
            actions.append(2)
        if self.agent_pos_row < 4:  # Can move down
            actions.append(3)
        return np.array(actions)

    def step(self, action: int):
        assert(not self.is_game_over())
        if action == 0:  
            self.agent_pos_col -= 1
        elif action == 1:
            self.agent_pos_col += 1
        elif action == 2:  
            self.agent_pos_row -= 1
        elif action == 3:
            self.agent_pos_row += 1

    def step_with_state_return(self, action: int) -> Tuple[int, int]:
        if self.is_game_over():
            return self.agent_pos_col, self.agent_pos_row
        self.step(action)
        return self.agent_pos_col, self.agent_pos_row

    def score(self) -> float:
        if self.agent_pos_row == 0 and self.agent_pos_col == 4:
            return -3.0
        if self.agent_pos_row == 4 and self.agent_pos_col == 4:
            return 1.0
        if self.agent_pos_row > 4 or self.agent_pos_row < 0 or self.agent_pos_col > 4 or self.agent_pos_col < 0:
            return -1.0
        return 0.0

    def display(self):
        for num_row in range(5):
            for num_col in range(5):
                print('X' if self.agent_pos_col == num_col and self.agent_pos_row == num_row else '_', end=' ')
            print()
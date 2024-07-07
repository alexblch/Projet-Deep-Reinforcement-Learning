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
    
    


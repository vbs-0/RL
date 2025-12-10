"""Q-Learning algorithm for discrete state/action spaces."""

import numpy as np
from typing import Tuple, Any
from collections import defaultdict


class QLearning:
    """Tabular Q-Learning algorithm."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 seed: int = 42):
        """Initialize Q-Learning agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate for Q-updates
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        np.random.seed(seed)
        
        # Q-table: maps (state, action) -> Q-value
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.visit_counts = defaultdict(lambda: np.zeros(action_size))
    
    def select_action(self, state: Any, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        state_key = str(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values = self.q_table[state_key]
        return np.argmax(q_values)
    
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """Update Q-value using Q-Learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = str(state)
        next_state_key = str(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning update
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
        self.visit_counts[state_key][action] += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, state: Any) -> np.ndarray:
        """Get Q-values for a state.
        
        Args:
            state: State
            
        Returns:
            Q-values for all actions
        """
        return self.q_table[str(state)].copy()
    
    def reset(self):
        """Reset the agent."""
        self.q_table.clear()
        self.visit_counts.clear()
        self.epsilon = 1.0

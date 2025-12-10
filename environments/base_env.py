"""Base environment class for all RL environments."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any


class BaseRLEnvironment(ABC):
    """Abstract base class for RL environments with standard interface."""
    
    def __init__(self, seed: int = 42):
        """Initialize environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: Next observation
            reward: Reward for the action
            done: Whether episode is done
            info: Additional info dict
        """
        pass
    
    @abstractmethod
    def render(self) -> Any:
        """Render the environment."""
        pass
    
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of observations."""
        pass
    
    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """Shape of actions."""
        pass
    
    def set_seed(self, seed: int):
        """Set random seed."""
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)

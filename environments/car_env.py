"""Car racing environment wrapper for RL experiments."""

import numpy as np
from typing import Tuple, Dict, Any
from .base_env import BaseRLEnvironment
import sys
sys.path.insert(0, '/workspaces/RL')

from test import Car, Track, BLACK


class CarRacingEnvironment(BaseRLEnvironment):
    """Car racing environment for RL agents."""
    
    def __init__(self, seed: int = 42, max_steps: int = 1000):
        """Initialize car racing environment.
        
        Args:
            seed: Random seed
            max_steps: Maximum steps per episode
        """
        super().__init__(seed)
        self.track = Track(seed=seed)
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car = Car(start_x, start_y)
        self.car.state = np.array([start_x, start_y, 0.0, start_angle])
        
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_reward = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.track = Track(seed=self.seed_value)
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car = Car(start_x, start_y)
        self.car.state = np.array([start_x, start_y, 0.0, start_angle])
        self.car.update_sensors(self.track.track_points, self.track.obstacles)
        self.current_step = 0
        self.episode_reward = 0
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step."""
        # Handle both continuous actions and discrete action indices
        if isinstance(action, (int, np.integer)):
            # For discrete actions, convert to continuous
            # Assuming 0 = left, 1 = straight, 2 = right, 3 = accelerate, etc.
            action_mapping = {
                0: np.array([-1.0, 0.0]),   # Left
                1: np.array([0.0, 0.0]),    # Straight
                2: np.array([1.0, 0.0]),    # Right
                3: np.array([0.0, 1.0]),    # Accelerate
                4: np.array([0.0, -1.0]),   # Brake
            }
            steering, throttle = action_mapping.get(action % 5, np.array([0.0, 0.0]))
        else:
            steering, throttle = float(action[0]), float(action[1])
        
        on_track = self.car.update([steering, throttle], self.track.track_points, self.track.obstacles)
        
        reward = self._calculate_reward(on_track)
        done = not on_track or self.current_step >= self.max_steps
        
        self.current_step += 1
        self.episode_reward += reward
        
        obs = self._get_observation()
        info = {'on_track': on_track, 'step': self.current_step, 'episode_reward': self.episode_reward}
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        sensors = np.array(self.car.sensor_readings) / self.car.sensor_length
        speed = float(self.car.state[2]) / self.car.max_speed
        obs = np.append(sensors, speed)
        return obs.astype(np.float32)
    
    def _calculate_reward(self, on_track: bool) -> float:
        """Calculate reward."""
        if not on_track:
            return -10.0
        
        speed_reward = float(self.car.state[2])
        progress = self.car.calculate_track_progress(self.track.track_points)
        progress_reward = 0.0
        
        if progress > self.car.last_checkpoint + self.track.checkpoint_interval:
            progress_reward = 5.0
            self.car.last_checkpoint = progress
        
        return speed_reward + progress_reward + 0.1
    
    def render(self):
        """Render environment."""
        pass
    
    @property
    def observation_shape(self) -> Tuple[int]:
        """Shape of observations: 5 sensors + 1 speed = 6."""
        return (6,)
    
    @property
    def action_shape(self) -> Tuple[int]:
        """Shape of actions: steering + throttle = 2."""
        return (2,)

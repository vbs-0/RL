"""Evaluation and metrics tools for RL experiments."""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import json
import os
from datetime import datetime


class Metrics:
    """Track and log training metrics."""
    
    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.
        
        Args:
            window_size: Moving average window size
        """
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.training_losses = deque(maxlen=window_size)
        self.exploration_rates = deque(maxlen=window_size)
        
        self.timestep = 0
        self.episode = 0
    
    def record_episode(self, reward: float, length: int):
        """Record episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode += 1
    
    def record_loss(self, loss: float):
        """Record training loss."""
        if loss is not None:
            self.training_losses.append(loss)
    
    def record_exploration(self, exploration_rate: float):
        """Record exploration rate."""
        self.exploration_rates.append(exploration_rate)
    
    def record_step(self):
        """Record a training step."""
        self.timestep += 1
    
    @property
    def mean_reward(self) -> Optional[float]:
        """Mean episode reward."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else None
    
    @property
    def max_reward(self) -> Optional[float]:
        """Max episode reward."""
        return float(np.max(self.episode_rewards)) if self.episode_rewards else None
    
    @property
    def mean_length(self) -> Optional[float]:
        """Mean episode length."""
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else None
    
    @property
    def mean_loss(self) -> Optional[float]:
        """Mean training loss."""
        return float(np.mean(self.training_losses)) if self.training_losses else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'timestep': self.timestep,
            'episode': self.episode,
            'mean_reward': self.mean_reward,
            'max_reward': self.max_reward,
            'mean_length': self.mean_length,
            'mean_loss': self.mean_loss,
            'num_episodes': len(self.episode_rewards)
        }
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert metrics to dictionary."""
        return {
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'training_losses': list(self.training_losses),
            'exploration_rates': list(self.exploration_rates)
        }


class ExperimentLogger:
    """Log experiment data to disk."""
    
    def __init__(self, experiment_name: str, log_dir: str = 'experiment_logs'):
        """Initialize logger.
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create log directory
        self.exp_dir = os.path.join(log_dir, f'{experiment_name}_{self.timestamp}')
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.exp_dir, 'metrics.json')
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_file = os.path.join(self.exp_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, metrics: Metrics):
        """Log metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def save_checkpoint(self, agent: Any, episode: int):
        """Save agent checkpoint."""
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f'agent_ep{episode}.pkl')
        
        import pickle
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(agent, f)
    
    def get_log_dir(self) -> str:
        """Get experiment log directory."""
        return self.exp_dir


class Evaluator:
    """Evaluate agent performance."""
    
    def __init__(self, env: Any, metrics: Metrics):
        """Initialize evaluator.
        
        Args:
            env: RL environment
            metrics: Metrics tracker
        """
        self.env = env
        self.metrics = metrics
    
    def evaluate(self, agent: Any, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """Evaluate agent for multiple episodes.
        
        Args:
            agent: Agent to evaluate
            num_episodes: Number of evaluation episodes
            render: Whether to render
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = agent.select_action(obs, training=False)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'max_reward': float(np.max(episode_rewards)),
            'min_reward': float(np.min(episode_rewards))
        }

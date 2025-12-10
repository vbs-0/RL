"""RL experiment pipeline with reproducibility and tracking."""

import numpy as np
from typing import Dict, Any, Optional, Type
import json
import os
from datetime import datetime

from algorithms import QLearning, DeepQNetwork, PolicyGradient, WorldModelAgent
from environments import CarRacingEnvironment
from evaluation import Metrics, ExperimentLogger, Evaluator


class RLExperiment:
    """Reproducible RL experiment pipeline."""
    
    def __init__(self,
                 algorithm_name: str,
                 env_config: Dict[str, Any],
                 algorithm_config: Dict[str, Any],
                 experiment_name: str = 'rl_experiment',
                 seed: int = 42):
        """Initialize experiment.
        
        Args:
            algorithm_name: Name of algorithm (qlearning, dqn, policy_gradient, world_model)
            env_config: Environment configuration
            algorithm_config: Algorithm configuration
            experiment_name: Name of experiment
            seed: Random seed
        """
        self.seed = seed
        np.random.seed(seed)
        
        self.algorithm_name = algorithm_name
        self.env_config = env_config
        self.algorithm_config = algorithm_config
        self.experiment_name = experiment_name
        
        # Initialize environment
        self.env = CarRacingEnvironment(seed=seed, **env_config)
        
        # Initialize algorithm
        self.agent = self._create_agent()
        
        # Initialize tracking
        self.metrics = Metrics()
        self.logger = ExperimentLogger(experiment_name)
        self.evaluator = Evaluator(self.env, self.metrics)
        
        # Log configuration
        config = {
            'algorithm': algorithm_name,
            'env_config': env_config,
            'algorithm_config': algorithm_config,
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.log_config(config)
    
    def _create_agent(self) -> Any:
        """Create agent based on algorithm name."""
        state_size = self.env.observation_shape[0]
        action_size = self.env.action_shape[0]
        
        if self.algorithm_name == 'qlearning':
            return QLearning(state_size, action_size, seed=self.seed, **self.algorithm_config)
        elif self.algorithm_name == 'dqn':
            return DeepQNetwork(state_size, action_size, seed=self.seed, **self.algorithm_config)
        elif self.algorithm_name == 'policy_gradient':
            return PolicyGradient(state_size, action_size, seed=self.seed, **self.algorithm_config)
        elif self.algorithm_name == 'world_model':
            return WorldModelAgent(state_size, action_size, seed=self.seed, **self.algorithm_config)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def run(self,
            num_episodes: int = 100,
            max_steps_per_episode: int = 1000,
            eval_interval: int = 10,
            eval_episodes: int = 5,
            save_interval: int = 50) -> Dict[str, Any]:
        """Run experiment.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Max steps per episode
            eval_interval: Evaluation interval (episodes)
            eval_episodes: Number of evaluation episodes
            save_interval: Checkpoint save interval (episodes)
            
        Returns:
            Final results
        """
        results = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and episode_length < max_steps_per_episode:
                # Select action
                action = self.agent.select_action(obs, training=True)
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Store transition (if agent supports it)
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(obs, action, reward, next_obs, done)
                
                # Train (if agent supports it)
                if hasattr(self.agent, 'train_step') and episode_length % 10 == 0:
                    loss = self.agent.train_step()
                    self.metrics.record_loss(loss)
                
                # Update
                if hasattr(self.agent, 'update'):
                    self.agent.update(obs, action, reward, next_obs, done)
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                self.metrics.record_step()
            
            # Record episode
            self.metrics.record_episode(episode_reward, episode_length)
            
            # Record exploration rate
            if hasattr(self.agent, 'epsilon'):
                self.metrics.record_exploration(self.agent.epsilon)
            
            # Periodic evaluation
            if (episode + 1) % eval_interval == 0:
                eval_results = self.evaluator.evaluate(self.agent, num_episodes=eval_episodes)
                eval_results['episode'] = episode + 1
                results.append(eval_results)
                print(f"Episode {episode + 1}: Mean Eval Reward={eval_results['mean_reward']:.2f}")
            
            # Save checkpoint (skip for JAX-based agents due to pickle limitations)
            if (episode + 1) % save_interval == 0:
                try:
                    self.logger.save_checkpoint(self.agent, episode + 1)
                except (AttributeError, TypeError):
                    # JAX agents cannot be pickled
                    pass
            
            # Periodic logging
            if (episode + 1) % 10 == 0:
                summary = self.metrics.get_summary()
                loss_str = f"{summary['mean_loss']:.4f}" if summary['mean_loss'] else 'N/A'
                print(f"Episode {episode + 1}: Mean Reward={summary['mean_reward']:.2f}, "
                      f"Loss={loss_str}")
        
        # Final evaluation
        print("\n=== Final Evaluation ===")
        final_eval = self.evaluator.evaluate(self.agent, num_episodes=20)
        print(f"Final Mean Reward: {final_eval['mean_reward']:.2f}")
        print(f"Final Std Reward: {final_eval['std_reward']:.2f}")
        print(f"Final Max Reward: {final_eval['max_reward']:.2f}")
        
        # Save metrics
        self.logger.log_metrics(self.metrics)
        
        return {
            'algorithm': self.algorithm_name,
            'final_eval': final_eval,
            'metrics': self.metrics.get_summary(),
            'log_dir': self.logger.get_log_dir()
        }

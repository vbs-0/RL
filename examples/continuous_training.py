#!/usr/bin/env python3
"""
Continuous Training Example - Headless Run

This script demonstrates continuous training of the RL agent in headless mode
(no display). It runs multiple episodes, trains the world model and actor,
and saves checkpoints when progress improves.

Usage:
    python examples/continuous_training.py [--episodes N] [--seed SEED]
"""
import argparse
import sys
import os

# Add parent directory to path to import test module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['JAX_DISABLE_JIT'] = '1'
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode

import pygame
import numpy as np
import jax.numpy as jnp
import pickle
from test import Car, Track, WorldModel, Actor


class HeadlessTrainer:
    """Headless trainer for continuous training without display."""
    
    def __init__(self, seed=42):
        # Initialize pygame in headless mode
        pygame.init()
        
        # Create environment components
        self.track = Track(seed=seed)
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car = Car(start_x, start_y)
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        
        # Initialize models
        obs_size = len(self.car.sensor_angles) + 1  # sensors + speed
        action_size = 2  # steering, throttle
        self.world_model = WorldModel(obs_size, action_size, seed=seed)
        self.actor = Actor(32, action_size, seed=seed)
        
        # Training parameters
        self.train_every = 10
        self.step_counter = 0
        self.explore_prob = 1.0
        self.min_explore_prob = 0.1
        self.explore_decay = 0.999
        
        # Statistics
        self.episode_rewards = []
        self.training_losses = []
        self.current_episode_reward = 0.0
        self.best_progress = 0.0
        
        # Model saving
        self.model_save_path = "models/"
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
    
    def get_observation(self):
        """Get current observation from car state."""
        norm_sensors = np.array(self.car.sensor_readings) / self.car.sensor_length
        speed = float(self.car.state[2]) / self.car.max_speed
        obs = np.append(norm_sensors, speed)
        return jnp.array(obs)
    
    def reset(self):
        """Reset environment to initial state."""
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        self.car.reset()
        self.car.update_sensors(self.track.track_points, self.track.obstacles)
        self.current_episode_reward = 0.0
        return self.get_observation()
    
    def calculate_reward(self, on_track):
        """Calculate reward for current state."""
        if not on_track:
            return -10.0
        
        # Reward for speed
        speed_reward = float(self.car.state[2])
        
        # Reward for progress
        progress = self.car.calculate_track_progress(self.track.track_points)
        progress_reward = 0.0
        
        if progress > self.car.last_checkpoint + self.track.checkpoint_interval:
            progress_reward = 5.0
            self.car.last_checkpoint = progress
            current_time = pygame.time.get_ticks()
            time_bonus = 1000 / max(1, current_time - self.car.checkpoint_time)
            progress_reward += time_bonus
            self.car.checkpoint_time = current_time
        
        # Survival reward
        survival_reward = 0.1
        
        return speed_reward + progress_reward + survival_reward
    
    def step(self, action):
        """Execute one environment step."""
        # Convert action format
        steering = float(action[0])
        throttle = (float(action[1]) + 1) / 2
        
        # Update car
        on_track = self.car.update([steering, throttle],
                                   self.track.track_points,
                                   self.track.obstacles)
        
        # Get observation and reward
        next_obs = self.get_observation()
        reward = self.calculate_reward(on_track)
        done = not on_track
        
        self.current_episode_reward += reward
        
        # Training
        if self.step_counter % self.train_every == 0:
            self.world_model.add_experience(
                self.get_observation(),
                jnp.array(action),
                next_obs,
                reward
            )
            
            # Train world model
            world_loss = self.world_model.train_step()
            if world_loss > 0:
                self.training_losses.append(world_loss)
            
            # Train actor
            if len(self.world_model.buffer) >= self.world_model.batch_size:
                batch_size = min(32, len(self.world_model.buffer))
                latent_batch = []
                reward_batch = []
                
                for i in range(batch_size):
                    exp = self.world_model.buffer[i]
                    latent = self.world_model.encode(exp[0])
                    latent_batch.append(latent)
                    reward_batch.append(float(exp[3]))
                
                latent_batch = jnp.array(latent_batch)
                reward_batch = jnp.array(reward_batch)
                actor_loss = self.actor.train_step(latent_batch, reward_batch)
        
        self.step_counter += 1
        self.explore_prob = max(self.min_explore_prob,
                               self.explore_prob * self.explore_decay)
        
        # Check for progress improvement and save
        current_progress = self.car.calculate_track_progress(self.track.track_points)
        if current_progress > self.best_progress:
            self.best_progress = current_progress
            self.save_checkpoint()
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
        
        return next_obs, reward, done
    
    def get_action(self, obs):
        """Get action from policy."""
        from jax import random
        latent = self.world_model.encode(obs)
        explore = random.uniform(self.world_model.key, shape=()) < self.explore_prob
        self.world_model.key, _ = random.split(self.world_model.key)
        action = self.actor.get_action(latent, explore=explore)
        return action
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        model_data = {
            'encoder_params': self.world_model.encoder_params,
            'dynamics_params': self.world_model.dynamics_params,
            'decoder_params': self.world_model.decoder_params,
            'reward_params': self.world_model.reward_params,
            'actor_params': self.actor.actor_params,
            'explore_prob': self.explore_prob,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses,
            'best_progress': self.best_progress,
            'step_counter': self.step_counter
        }
        
        filename = f"{self.model_save_path}model_ep{len(self.episode_rewards)}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  → Checkpoint saved: {filename}")
    
    def train(self, num_episodes=10, max_steps_per_episode=1000):
        """Run continuous training for specified number of episodes."""
        print(f"Starting continuous training for {num_episodes} episodes...")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print(f"Models will be saved to: {self.model_save_path}")
        print("-" * 60)
        
        for episode in range(num_episodes):
            obs = self.reset()
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # Get action from policy
                action = self.get_action(obs)
                
                # Execute step
                next_obs, reward, done = self.step(action)
                obs = next_obs
                episode_steps += 1
                
                if done:
                    break
            
            # Episode summary
            episode_reward = self.episode_rewards[-1] if self.episode_rewards else 0
            avg_loss = np.mean(self.training_losses[-10:]) if self.training_losses else 0
            
            print(f"Episode {episode + 1}/{num_episodes}:")
            print(f"  Steps: {episode_steps}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Exploration: {self.explore_prob:.3f}")
            print(f"  Best Progress: {self.best_progress:.3f}")
            print(f"  Avg Loss (last 10): {avg_loss:.4f}")
            print(f"  Buffer Size: {len(self.world_model.buffer)}")
        
        print("-" * 60)
        print("Training complete!")
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Total steps: {self.step_counter}")
        print(f"Best progress achieved: {self.best_progress:.3f}")
        
        # Save final model
        self.save_checkpoint()
        print(f"Final model saved.")


def main():
    parser = argparse.ArgumentParser(description='Continuous training example')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to train (default: 10)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Max steps per episode (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = HeadlessTrainer(seed=args.seed)
    trainer.train(num_episodes=args.episodes, max_steps_per_episode=args.max_steps)


if __name__ == '__main__':
    main()

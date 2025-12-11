"""Training module containing the ContinuousTrainer and training orchestration logic."""

import jax
import jax.numpy as jnp
from jax import random
import pygame
import numpy as np
import os
import pickle
import time
from typing import Dict, Any, Optional, Tuple
import threading
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_system.env import RLCarEnv
from rl_system.agents import WorldModel, Actor
from rl_system.config import config


class ContinuousTrainer:
    """Continuous trainer that manages the training loop for both CLI and web interfaces."""
    
    def __init__(self, config_path: Optional[str] = None, headless: bool = False):
        """Initialize the continuous trainer."""
        # Load configuration
        if config_path:
            config.load_config(config_path)
        
        # Initialize environment and agents
        self.env = RLCarEnv()
        
        # Get observation and action sizes
        obs_size = len(self.env.car.sensor_angles) + 1  # sensors + speed
        action_size = 2  # steering, throttle
        latent_size = config.get('agent.latent_size', 32)
        
        # Initialize world model and actor
        self.world_model = WorldModel(obs_size, action_size)
        self.actor = Actor(latent_size, action_size)
        
        # Connect environment to agents
        self.env.world_model = self.world_model
        self.env.actor = self.actor
        
        # Training state
        self.step_counter = 0
        self.episode_counter = 0
        self.explore_prob = config.get('training.exploration_initial', 1.0)
        self.min_explore_prob = config.get('training.exploration_min', 0.1)
        self.explore_decay = config.get('training.exploration_decay', 0.999)
        self.train_every = config.get('training.train_every', 10)
        self.save_interval = config.get('training.save_interval', 10)
        
        # Statistics
        self.episode_rewards = []
        self.training_losses = []
        self.world_losses = []
        self.actor_losses = []
        self.current_episode_reward = 0
        self.best_progress = 0.0
        
        # Model saving
        self.model_save_path = config.get('model.save_path', "models/")
        self.filename_prefix = config.get('model.filename_prefix', "model_ep")
        self.current_model_path = None
        
        # Control flags
        self.running = False
        self.manual_control = False
        self.current_action = [0.0, 0.0]  # [steering, throttle]
        self.headless = headless
        
        # Initialize display if not headless
        if not self.headless:
            pygame.init()
            width = config.get('environment.width', 800)
            height = config.get('environment.height', 600)
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("RL Car Game with DreamerV3 - Continuous Training")
            self.clock = pygame.time.Clock()
            self.fps = config.get('environment.fps', 60)
        else:
            # Headless mode setup
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            width = config.get('environment.width', 800)
            height = config.get('environment.height', 600)
            self.screen = pygame.Surface((width, height))
            self.clock = pygame.time.Clock()
            self.fps = config.get('environment.fps', 60)
        
        # Create model directory
        os.makedirs(self.model_save_path, exist_ok=True)
        
        print(f"ContinuousTrainer initialized - Latent size: {latent_size}, Action size: {action_size}")
    
    def get_action(self, obs: jnp.ndarray, explore: bool = None) -> jnp.ndarray:
        """Get action for given observation."""
        if explore is None:
            explore = random.uniform(self._get_key(), shape=()) < self.explore_prob
        
        latent = self.world_model.encode(obs)
        action = self.actor.get_action(latent, explore=explore)
        return action
    
    def _get_key(self):
        """Get a random key for exploration."""
        if not hasattr(self, 'key'):
            self.key = random.PRNGKey(42)
        self.key, subkey = random.split(self.key)
        return subkey
    
    def train_step(self, obs: jnp.ndarray, action: jnp.ndarray, 
                   next_obs: jnp.ndarray, reward: float) -> Dict[str, float]:
        """Perform one training step."""
        # Add experience to replay buffer
        self.world_model.add_experience(obs, action, next_obs, reward)
        
        losses = {}
        
        # Train world model
        if self.step_counter % self.train_every == 0 and len(self.world_model.buffer) >= self.world_model.batch_size:
            world_loss = self.world_model.train_step()
            self.training_losses.append(world_loss)
            losses['world_loss'] = world_loss
            
            # Train actor on recent batch (disabled due to JAX/Optax issues)
            # TODO: Re-enable actor training once JAX/Optax compatibility is resolved
            # if len(self.world_model.buffer) >= self.world_model.batch_size:
            #     batch_size = min(self.world_model.batch_size, len(self.world_model.buffer))
            #     indices = random.randint(self._get_key(), (batch_size,), 0, len(self.world_model.buffer))
            #     batch = [self.world_model.buffer[int(i)] for i in indices]
            #     
            #     latent_batch = jnp.array([self.world_model.encode(item[0]) for item in batch])
            #     reward_batch = jnp.array([float(item[3]) for item in batch])
            #     
            #     actor_loss = self.actor.train_step(latent_batch, reward_batch)
            #     self.actor_losses.append(actor_loss)
            #     losses['actor_loss'] = actor_loss
        
        # Update exploration probability
        self.explore_prob = max(self.min_explore_prob, self.explore_prob * self.explore_decay)
        
        # Update step counter
        self.step_counter += 1
        
        return losses
    
    def check_and_save_model(self) -> bool:
        """Check progress and save model if improved. Returns True if saved."""
        current_progress = self.env.car.calculate_track_progress(self.env.track.track_points)
        
        if current_progress > self.best_progress:
            self.best_progress = current_progress
            self.save_model()
            return True
        return False
    
    def save_model(self, episode: Optional[int] = None):
        """Save current model state."""
        if episode is None:
            episode = self.episode_counter
        
        filename = f"{self.filename_prefix}{episode}.pkl"
        filepath = os.path.join(self.model_save_path, filename)
        
        model_data = {
            'episode': episode,
            'step': self.step_counter,
            'world_model_params': {
                'encoder_params': self.world_model.encoder_params,
                'dynamics_params': self.world_model.dynamics_params,
                'decoder_params': self.world_model.decoder_params,
                'reward_params': self.world_model.reward_params,
                'encoder_opt_state': self.world_model.encoder_opt_state,
                'dynamics_opt_state': self.world_model.dynamics_opt_state,
                'decoder_opt_state': self.world_model.decoder_opt_state,
                'reward_opt_state': self.world_model.reward_opt_state,
            },
            'actor_params': {
                'actor_params': self.actor.actor_params,
                'actor_opt_state': self.actor.actor_opt_state,
            },
            'training_state': {
                'explore_prob': self.explore_prob,
                'episode_rewards': self.episode_rewards,
                'training_losses': self.training_losses,
                'world_losses': self.world_losses,
                'actor_losses': self.actor_losses,
                'best_progress': self.best_progress,
                'step_counter': self.step_counter,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.current_model_path = filepath
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """Load model state from file."""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load world model parameters
            wm_params = model_data.get('world_model_params', {})
            self.world_model.encoder_params = wm_params.get('encoder_params', self.world_model.encoder_params)
            self.world_model.dynamics_params = wm_params.get('dynamics_params', self.world_model.dynamics_params)
            self.world_model.decoder_params = wm_params.get('decoder_params', self.world_model.decoder_params)
            self.world_model.reward_params = wm_params.get('reward_params', self.world_model.reward_params)
            
            # Load optimizer states
            self.world_model.encoder_opt_state = wm_params.get('encoder_opt_state', self.world_model.encoder_opt_state)
            self.world_model.dynamics_opt_state = wm_params.get('dynamics_opt_state', self.world_model.dynamics_opt_state)
            self.world_model.decoder_opt_state = wm_params.get('decoder_opt_state', self.world_model.decoder_opt_state)
            self.world_model.reward_opt_state = wm_params.get('reward_opt_state', self.world_model.reward_opt_state)
            
            # Load actor parameters
            actor_params = model_data.get('actor_params', {})
            self.actor.actor_params = actor_params.get('actor_params', self.actor.actor_params)
            self.actor.actor_opt_state = actor_params.get('actor_opt_state', self.actor.actor_opt_state)
            
            # Load training state
            training_state = model_data.get('training_state', {})
            self.explore_prob = training_state.get('explore_prob', self.explore_prob)
            self.episode_rewards = training_state.get('episode_rewards', [])
            self.training_losses = training_state.get('training_losses', [])
            self.world_losses = training_state.get('world_losses', [])
            self.actor_losses = training_state.get('actor_losses', [])
            self.best_progress = training_state.get('best_progress', 0.0)
            self.step_counter = training_state.get('step_counter', 0)
            
            # Update episode counter
            self.episode_counter = model_data.get('episode', 0)
            
            print(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'episode': self.episode_counter,
            'step': self.step_counter,
            'exploration_rate': float(self.explore_prob),
            'episode_rewards': self.episode_rewards[-100:],  # Last 100 episodes
            'training_losses': self.training_losses[-100:],  # Last 100 steps
            'world_losses': self.world_losses[-100:],
            'actor_losses': self.actor_losses[-100:],
            'best_progress': float(self.best_progress),
            'buffer_size': len(self.world_model.buffer),
            'running': self.running,
            'manual_control': self.manual_control
        }
    
    def list_models(self) -> list:
        """List all saved models."""
        models = []
        if os.path.exists(self.model_save_path):
            models = [f for f in os.listdir(self.model_save_path) if f.endswith('.pkl')]
            models.sort()
        return models
    
    def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run a single episode."""
        obs = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Get action
            if self.manual_control:
                action = jnp.array(self.current_action)
            else:
                action = self.get_action(obs)
            
            # Step environment
            next_obs, reward, done = self.env.step(np.array(action))
            episode_reward += reward
            episode_steps += 1
            
            # Train
            losses = self.train_step(obs, action, next_obs, reward)
            
            obs = next_obs
            
            # Update episode tracking
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_counter += 1
                
                # Check for model saving
                saved = self.check_and_save_model()
                
                # Print episode summary
                print(f"Episode {self.episode_counter}: Reward={episode_reward:.2f}, Steps={episode_steps}, "
                      f"Progress={self.env.car.calculate_track_progress(self.env.track.track_points):.3f}, "
                      f"Explore={self.explore_prob:.3f}, Saved={saved}")
                
                return {
                    'episode': self.episode_counter,
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'progress': self.env.car.calculate_track_progress(self.env.track.track_points),
                    'losses': losses,
                    'done': True
                }
        
        # Episode didn't finish naturally
        self.episode_rewards.append(episode_reward)
        self.episode_counter += 1
        
        return {
            'episode': self.episode_counter,
            'reward': episode_reward,
            'steps': episode_steps,
            'progress': self.env.car.calculate_track_progress(self.env.track.track_points),
            'losses': losses,
            'done': False,
            'max_steps_reached': True
        }
    
    def run_continuous(self, max_episodes: Optional[int] = None, render: bool = None):
        """Run continuous training loop."""
        if render is None:
            render = not self.headless
        
        self.running = True
        episode = 0
        
        print(f"Starting continuous training (max_episodes={max_episodes}, render={render})")
        
        try:
            while self.running:
                if max_episodes and episode >= max_episodes:
                    print(f"Reached maximum episodes: {max_episodes}")
                    break
                
                # Run single episode
                result = self.run_episode()
                episode += 1
                
                # Render if enabled
                if render and not self.headless:
                    self.render()
                    
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                self.manual_control = not self.manual_control
                            elif event.key == pygame.K_s:
                                self.save_model()
                            elif event.key == pygame.K_ESCAPE:
                                self.running = False
                    
                    # Control frame rate
                    self.clock.tick(self.fps)
        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            self.running = False
            print(f"Training completed. Total episodes: {self.episode_counter}")
    
    def render(self):
        """Render the current state."""
        if not self.headless:
            self.env.render(self.screen)
            
            # Add training info overlay
            font = pygame.font.SysFont(None, 24)
            colors = config.get('colors', {})
            white = tuple(colors.get('white', [255, 255, 255]))
            
            # Training stats
            stats_text = f"Episode: {self.episode_counter} | Step: {self.step_counter} | "
            stats_text += f"Explore: {self.explore_prob:.3f} | "
            stats_text += f"Buffer: {len(self.world_model.buffer)}"
            
            if self.manual_control:
                control_text = "MANUAL CONTROL (Space to toggle)"
            else:
                control_text = "AI CONTROL (Space to toggle)"
            
            stats_surface = font.render(stats_text, True, white)
            control_surface = font.render(control_text, True, white)
            
            self.screen.blit(stats_surface, (10, config.get('environment.height', 600) - 60))
            self.screen.blit(control_surface, (10, config.get('environment.height', 600) - 30))
            
            pygame.display.flip()
    
    def stop(self):
        """Stop the training loop."""
        self.running = False
        print("Stopping training...")
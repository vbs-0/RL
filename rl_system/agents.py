"""Agents module containing WorldModel and Actor classes."""

import jax
import jax.numpy as jnp
from jax import random, grad
import haiku as hk
import optax
from collections import deque
import numpy as np
from typing import Tuple, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_system.config import config


class WorldModel:
    """Dreamer-style world model with encoder, dynamics, decoder, and reward networks."""
    
    def __init__(self, obs_size: int, action_size: int, seed: int = 42):
        """Initialize world model."""
        self.obs_size = obs_size
        self.action_size = action_size
        self.latent_size = config.get('agent.latent_size', 32)
        self.key = random.PRNGKey(seed)
        
        # Network configurations
        self.encoder_hidden = config.get('world_model.encoder_hidden', 64)
        self.dynamics_hidden = config.get('world_model.dynamics_hidden', 64)
        self.decoder_hidden = config.get('world_model.decoder_hidden', 64)
        self.reward_hidden = config.get('world_model.reward_hidden', 32)
        self.learning_rate = config.get('world_model.learning_rate', 1e-3)
        
        # Build networks
        self._build_networks()
        
        # Experience replay buffer
        buffer_size = config.get('training.buffer_size', 10000)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = config.get('training.batch_size', 32)
    
    def _build_networks(self):
        """Build and initialize all world model networks."""
        # Define network architectures
        def encoder_fn(obs):
            net = hk.Sequential([
                hk.Linear(self.encoder_hidden), jax.nn.relu,
                hk.Linear(self.latent_size)
            ])
            return net(obs)
        
        def dynamics_fn(latent, action):
            net = hk.Sequential([
                hk.Linear(self.dynamics_hidden), jax.nn.relu,
                hk.Linear(self.latent_size)
            ])
            x = jnp.concatenate([latent, action], axis=-1)
            return net(x)
        
        def decoder_fn(latent):
            net = hk.Sequential([
                hk.Linear(self.decoder_hidden), jax.nn.relu,
                hk.Linear(self.obs_size)
            ])
            return net(latent)
        
        def reward_fn(latent):
            net = hk.Sequential([
                hk.Linear(self.reward_hidden), jax.nn.relu,
                hk.Linear(1)
            ])
            return net(latent)
        
        # Transform into pure functions
        self.encoder = hk.transform(encoder_fn)
        self.dynamics = hk.transform(dynamics_fn)
        self.decoder = hk.transform(decoder_fn)
        self.reward = hk.transform(reward_fn)
        
        # Initialize parameters
        dummy_obs = jnp.zeros((1, self.obs_size))
        dummy_latent = jnp.zeros((1, self.latent_size))
        dummy_action = jnp.zeros((1, self.action_size))
        
        self.key, subkey1, subkey2, subkey3, subkey4 = random.split(self.key, 5)
        self.encoder_params = self.encoder.init(subkey1, dummy_obs)
        self.dynamics_params = self.dynamics.init(subkey2, dummy_latent, dummy_action)
        self.decoder_params = self.decoder.init(subkey3, dummy_latent)
        self.reward_params = self.reward.init(subkey4, dummy_latent)
        
        # Initialize optimizers
        self.encoder_optimizer = optax.adam(self.learning_rate)
        self.dynamics_optimizer = optax.adam(self.learning_rate)
        self.decoder_optimizer = optax.adam(self.learning_rate)
        self.reward_optimizer = optax.adam(self.learning_rate)
        
        # Initialize optimizer states
        self.encoder_opt_state = self.encoder_optimizer.init(self.encoder_params)
        self.dynamics_opt_state = self.dynamics_optimizer.init(self.dynamics_params)
        self.decoder_opt_state = self.decoder_optimizer.init(self.decoder_params)
        self.reward_opt_state = self.reward_optimizer.init(self.reward_params)
    
    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation to latent representation."""
        obs_batch = jnp.expand_dims(obs, axis=0)
        return self.encoder.apply(self.encoder_params, None, obs_batch)[0]
    
    def predict_next(self, latent: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """Predict next latent state and reward."""
        latent_batch = jnp.expand_dims(latent, axis=0)
        action_batch = jnp.expand_dims(action, axis=0)
        
        next_latent = self.dynamics.apply(self.dynamics_params, None, latent_batch, action_batch)[0]
        predicted_reward = float(self.reward.apply(self.reward_params, None, latent_batch)[0])
        
        return next_latent, predicted_reward
    
    def decode(self, latent: jnp.ndarray) -> jnp.ndarray:
        """Decode latent state to observation."""
        latent_batch = jnp.expand_dims(latent, axis=0)
        return self.decoder.apply(self.decoder_params, None, latent_batch)[0]
    
    def add_experience(self, obs: jnp.ndarray, action: jnp.ndarray, 
                      next_obs: jnp.ndarray, reward: float):
        """Add experience to replay buffer."""
        self.buffer.append((obs, action, next_obs, reward))
    
    def train_step(self) -> float:
        """Train world model on a batch of experiences."""
        if len(self.buffer) < self.batch_size:
            return 0.0  # Not enough data
        
        # Sample batch
        indices = random.randint(self.key, (self.batch_size,), 0, len(self.buffer))
        self.key, _ = random.split(self.key)
        
        batch = [self.buffer[int(i)] for i in indices]
        obs_batch = jnp.array([np.array(item[0]) for item in batch])
        action_batch = jnp.array([np.array(item[1]) for item in batch])
        next_obs_batch = jnp.array([np.array(item[2]) for item in batch])
        reward_batch = jnp.array([float(item[3]) for item in batch])
        
        # Compute total loss for monitoring
        latent_batch = self.encoder.apply(self.encoder_params, None, obs_batch)
        next_latent_batch = self.dynamics.apply(self.dynamics_params, None, latent_batch, action_batch)
        predicted_reward = self.reward.apply(self.reward_params, None, latent_batch)
        decoded_obs = self.decoder.apply(self.decoder_params, None, latent_batch)
        decoded_next_obs = self.decoder.apply(self.decoder_params, None, next_latent_batch)
        
        reconstruction_loss = jnp.mean((decoded_obs - obs_batch) ** 2)
        next_reconstruction_loss = jnp.mean((decoded_next_obs - next_obs_batch) ** 2)
        reward_loss = jnp.mean((predicted_reward.squeeze() - reward_batch) ** 2)
        total_loss = float(reconstruction_loss + next_reconstruction_loss + reward_loss)
        
        # For now, just compute and return loss without training
        # TODO: Fix gradient computation and training
        print(f"WorldModel Loss: {total_loss:.6f} (Reconstruction: {float(reconstruction_loss):.6f})")
        
        return total_loss
    
    def save_params(self, filepath: str):
        """Save model parameters."""
        import pickle
        model_data = {
            'encoder_params': self.encoder_params,
            'dynamics_params': self.dynamics_params,
            'decoder_params': self.decoder_params,
            'reward_params': self.reward_params,
            'encoder_opt_state': self.encoder_opt_state,
            'dynamics_opt_state': self.dynamics_opt_state,
            'decoder_opt_state': self.decoder_opt_state,
            'reward_opt_state': self.reward_opt_state,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_params(self, filepath: str):
        """Load model parameters."""
        import pickle
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.encoder_params = model_data.get('encoder_params', self.encoder_params)
            self.dynamics_params = model_data.get('dynamics_params', self.dynamics_params)
            self.decoder_params = model_data.get('decoder_params', self.decoder_params)
            self.reward_params = model_data.get('reward_params', self.reward_params)
            
            # Load optimizer states if available
            if 'encoder_opt_state' in model_data:
                self.encoder_opt_state = model_data['encoder_opt_state']
            if 'dynamics_opt_state' in model_data:
                self.dynamics_opt_state = model_data['dynamics_opt_state']
            if 'decoder_opt_state' in model_data:
                self.decoder_opt_state = model_data['decoder_opt_state']
            if 'reward_opt_state' in model_data:
                self.reward_opt_state = model_data['reward_opt_state']


class Actor:
    """Policy network that maps latent states to actions."""
    
    def __init__(self, latent_size: int, action_size: int, seed: int = 42):
        """Initialize actor network."""
        self.latent_size = latent_size
        self.action_size = action_size
        self.key = random.PRNGKey(seed)
        
        # Network configuration
        hidden_layers = config.get('actor.hidden_layers', [64, 32])
        self.learning_rate = config.get('actor.learning_rate', 1e-3)
        
        # Define actor network
        def actor_fn(latent):
            layers = []
            for hidden_size in hidden_layers:
                layers.extend([hk.Linear(hidden_size), jax.nn.relu])
            layers.append(hk.Linear(action_size))
            layers.append(jax.nn.tanh)
            
            net = hk.Sequential(layers)
            return net(latent)
        
        # Transform into pure function
        self.actor = hk.transform(actor_fn)
        
        # Initialize parameters
        dummy_latent = jnp.zeros((1, self.latent_size))
        self.key, subkey = random.split(self.key)
        self.actor_params = self.actor.init(subkey, dummy_latent)
        
        # Initialize optimizer
        self.actor_optimizer = optax.adam(self.learning_rate)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
    
    def get_action(self, latent: jnp.ndarray, explore: bool = False) -> jnp.ndarray:
        """Get action from latent state."""
        latent_batch = jnp.expand_dims(latent, axis=0)
        action = self.actor.apply(self.actor_params, None, latent_batch)[0]
        
        # Add exploration noise if needed
        if explore:
            self.key, subkey = random.split(self.key)
            noise = random.normal(subkey, shape=action.shape) * 0.2
            action = jnp.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def train_step(self, latent_batch: jnp.ndarray, reward_batch: jnp.ndarray) -> float:
        """Train actor using policy gradient."""
        def compute_actor_loss(actor_params):
            # Get actions from policy
            actions = self.actor.apply(actor_params, None, latent_batch)
            
            # Policy gradient loss: -mean(reward * action_output)
            actor_loss = -jnp.mean(reward_batch.reshape(-1, 1) * actions)
            return actor_loss
        
        # Compute gradient
        actor_grads = grad(compute_actor_loss)(self.actor_params)
        
        # Update parameters
        actor_updates, self.actor_opt_state = self.actor_optimizer.update(
            actor_grads, self.actor_opt_state
        )
        self.actor_params = optax.apply_updates(self.actor_params, actor_updates)
        
        return float(compute_actor_loss(self.actor_params))
    
    def save_params(self, filepath: str):
        """Save actor parameters."""
        import pickle
        model_data = {
            'actor_params': self.actor_params,
            'actor_opt_state': self.actor_opt_state,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_params(self, filepath: str):
        """Load actor parameters."""
        import pickle
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.actor_params = model_data.get('actor_params', self.actor_params)
            if 'actor_opt_state' in model_data:
                self.actor_opt_state = model_data['actor_opt_state']
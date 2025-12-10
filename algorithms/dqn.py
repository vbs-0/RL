"""Deep Q-Network (DQN) algorithm."""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad
import haiku as hk
import optax
from typing import Tuple, Any, Optional
from collections import deque


class DeepQNetwork:
    """Deep Q-Network (DQN) algorithm."""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 seed: int = 42):
        """Initialize DQN agent.
        
        Args:
            state_size: State observation size
            action_size: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Update target network frequency
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.key = random.PRNGKey(seed)
        
        # Define Q-network
        def q_network(state):
            net = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(action_size)
            ])
            return net(state)
        
        self.q_net = hk.transform(q_network)
        
        # Initialize parameters
        dummy_state = jnp.zeros((1, state_size))
        self.key, subkey = random.split(self.key)
        self.q_params = self.q_net.init(subkey, dummy_state)
        self.target_q_params = self.q_params
        
        # Optimizer
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.optimizer.init(self.q_params)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.train_step_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state_batch = jnp.expand_dims(state, axis=0)
        q_values = self.q_net.apply(self.q_params, None, state_batch)[0]
        return int(np.argmax(q_values))
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Perform one training step.
        
        Returns:
            Loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = jnp.array([item[0] for item in batch])
        actions = jnp.array([item[1] for item in batch])
        rewards = jnp.array([item[2] for item in batch])
        next_states = jnp.array([item[3] for item in batch])
        dones = jnp.array([item[4] for item in batch])
        
        # Define loss
        def loss_fn(params):
            # Current Q-values
            q_values = self.q_net.apply(params, None, states)
            q_selected = jnp.take_along_axis(q_values, actions[:, None], axis=1)
            
            # Target Q-values
            next_q_values = self.q_net.apply(self.target_q_params, None, next_states)
            max_next_q = jnp.max(next_q_values, axis=1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
            
            # MSE loss
            loss = jnp.mean((q_selected.squeeze() - target_q) ** 2)
            return loss
        
        # Update
        loss = loss_fn(self.q_params)
        grads = grad(loss_fn)(self.q_params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.q_params = optax.apply_updates(self.q_params, updates)
        
        self.train_step_count += 1
        
        # Update target network
        if self.train_step_count % self.target_update_freq == 0:
            self.target_q_params = self.q_params
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return float(loss)

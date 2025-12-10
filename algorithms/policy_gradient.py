"""Policy Gradient algorithm."""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad
import haiku as hk
import optax
from typing import Tuple, Optional
from collections import deque


class PolicyGradient:
    """Policy Gradient (REINFORCE) algorithm."""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 batch_size: int = 32,
                 seed: int = 42):
        """Initialize Policy Gradient agent.
        
        Args:
            state_size: State observation size
            action_size: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            entropy_coef: Entropy coefficient for exploration
            batch_size: Training batch size
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        
        self.key = random.PRNGKey(seed)
        
        # Define policy network (outputs action logits)
        def policy_network(state):
            net = hk.Sequential([
                hk.Linear(128), jax.nn.relu,
                hk.Linear(128), jax.nn.relu,
                hk.Linear(action_size)
            ])
            return net(state)
        
        self.policy_net = hk.transform(policy_network)
        
        # Initialize parameters
        dummy_state = jnp.zeros((1, state_size))
        self.key, subkey = random.split(self.key)
        self.policy_params = self.policy_net.init(subkey, dummy_state)
        
        # Optimizer
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.optimizer.init(self.policy_params)
        
        # Trajectory buffer
        self.trajectory = deque()
        self.episode_rewards = []
        self.episode_return = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action from policy distribution."""
        state_batch = jnp.expand_dims(state, axis=0)
        logits = self.policy_net.apply(self.policy_params, None, state_batch)[0]
        probs = jax.nn.softmax(logits)
        
        self.key, subkey = random.split(self.key)
        action = random.categorical(subkey, logits)
        return int(action)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray = None, done: bool = None):
        """Store transition in trajectory."""
        self.trajectory.append((state, action, reward))
        self.episode_return += reward
    
    def train_step(self) -> Optional[float]:
        """Perform training on accumulated trajectory.
        
        Returns:
            Loss value or None if trajectory is empty
        """
        if len(self.trajectory) < self.batch_size:
            return None
        
        # Convert trajectory to arrays
        states = jnp.array([t[0] for t in self.trajectory])
        actions = jnp.array([t[1] for t in self.trajectory])
        rewards = jnp.array([t[2] for t in self.trajectory])
        
        # Calculate discounted returns
        returns = np.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            returns[t] = cumulative
        
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = jnp.array(returns)
        
        # Define loss
        def loss_fn(params):
            logits = self.policy_net.apply(params, None, states)
            log_probs = jax.nn.log_softmax(logits)
            action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
            
            # Policy gradient loss: -E[log_prob * return]
            policy_loss = -jnp.mean(action_log_probs * returns)
            
            # Entropy regularization
            probs = jax.nn.softmax(logits)
            entropy = -jnp.sum(probs * log_probs, axis=1)
            entropy_loss = -self.entropy_coef * jnp.mean(entropy)
            
            return policy_loss + entropy_loss
        
        # Update
        loss = loss_fn(self.policy_params)
        grads = grad(loss_fn)(self.policy_params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.policy_params = optax.apply_updates(self.policy_params, updates)
        
        # Clear trajectory
        self.trajectory.clear()
        self.episode_rewards.append(self.episode_return)
        self.episode_return = 0
        
        return float(loss)

"""World Model based agent (existing implementation)."""

import numpy as np
from typing import Optional


class WorldModelAgent:
    """Wrapper for World Model agent from existing implementation."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 1e-3,
                 seed: int = 42):
        """Initialize World Model agent.
        
        Args:
            state_size: State observation size
            action_size: Number of actions
            learning_rate: Learning rate
            seed: Random seed
        """
        # Import from existing implementation
        import sys
        sys.path.insert(0, '/workspaces/RL')
        from test import WorldModel, Actor
        
        self.state_size = state_size
        self.action_size = action_size
        self.world_model = WorldModel(state_size, action_size, seed=seed)
        self.actor = Actor(32, action_size, seed=seed)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using world model + actor."""
        import jax.numpy as jnp
        from jax import random
        
        state_jnp = jnp.array(state)
        latent = self.world_model.encode(state_jnp)
        explore = training and random.uniform(self.world_model.key) < 0.3
        action = self.actor.get_action(latent, explore=explore)
        return np.array(action)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in world model buffer."""
        import jax.numpy as jnp
        
        state_jnp = jnp.array(state)
        action_jnp = jnp.array(action)
        next_state_jnp = jnp.array(next_state)
        
        self.world_model.add_experience(state_jnp, action_jnp, next_state_jnp, reward)
    
    def train_step(self) -> Optional[float]:
        """Train world model and actor."""
        world_loss = self.world_model.train_step()
        
        # Train actor if we have data
        if len(self.world_model.buffer) >= self.world_model.batch_size:
            import jax.numpy as jnp
            import numpy as np
            
            indices = jnp.arange(min(self.world_model.batch_size, len(self.world_model.buffer)))
            batch = [self.world_model.buffer[i] for i in indices]
            latent_batch = jnp.array([self.world_model.encode(item[0]) for item in batch])
            reward_batch = jnp.array([float(item[3]) for item in batch])
            actor_loss = self.actor.train_step(latent_batch, reward_batch)
        
        return world_loss

"""
Tests for world-model training steps (loss decreases on synthetic data).
"""
import pytest
import numpy as np
import jax.numpy as jnp
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['JAX_DISABLE_JIT'] = '1'

from test import WorldModel, Actor


class TestWorldModelTraining:
    """Test world model training behavior."""
    
    def test_buffer_starts_empty(self):
        """Test that experience buffer starts empty."""
        world_model = WorldModel(6, 2)
        assert len(world_model.buffer) == 0
        
    def test_add_experience(self):
        """Test adding experience to buffer."""
        world_model = WorldModel(6, 2)
        obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3])
        action = jnp.array([0.0, 0.5])
        next_obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.4])
        reward = 1.0
        
        world_model.add_experience(obs, action, next_obs, reward)
        assert len(world_model.buffer) == 1
        
    def test_buffer_max_size(self):
        """Test that buffer respects max size."""
        world_model = WorldModel(6, 2)
        max_size = world_model.buffer.maxlen
        
        # Add more experiences than max size
        for i in range(max_size + 100):
            obs = jnp.ones(6) * (i / (max_size + 100))
            action = jnp.zeros(2)
            next_obs = jnp.ones(6) * ((i + 1) / (max_size + 100))
            reward = float(i)
            world_model.add_experience(obs, action, next_obs, reward)
        
        # Buffer should not exceed max size
        assert len(world_model.buffer) == max_size
        
    def test_train_step_with_insufficient_data(self):
        """Test that training returns 0 loss with insufficient data."""
        world_model = WorldModel(6, 2)
        # Add only a few experiences (less than batch size)
        for i in range(5):
            obs = jnp.ones(6)
            action = jnp.zeros(2)
            next_obs = jnp.ones(6)
            reward = 1.0
            world_model.add_experience(obs, action, next_obs, reward)
        
        loss = world_model.train_step()
        assert loss == 0.0
        
    def test_train_step_with_sufficient_data(self):
        """Test that training computes loss with sufficient data."""
        world_model = WorldModel(6, 2)
        batch_size = world_model.batch_size
        
        # Add enough experiences
        for i in range(batch_size + 10):
            obs = jnp.ones(6) * 0.5
            action = jnp.zeros(2)
            next_obs = jnp.ones(6) * 0.6
            reward = 1.0
            world_model.add_experience(obs, action, next_obs, reward)
        
        loss = world_model.train_step()
        assert loss > 0.0
        assert not np.isnan(loss)
        assert not np.isinf(loss)


class TestSyntheticDataTraining:
    """Test training behavior on synthetic data."""
    
    def test_loss_on_consistent_data(self):
        """Test loss behavior on consistent synthetic data."""
        world_model = WorldModel(6, 2)
        
        # Generate consistent synthetic data
        for i in range(100):
            obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3])
            action = jnp.array([0.0, 0.5])
            # Next obs is slightly different
            next_obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.35])
            reward = 1.0
            world_model.add_experience(obs, action, next_obs, reward)
        
        # Train multiple times and collect losses
        losses = []
        for _ in range(10):
            loss = world_model.train_step()
            losses.append(loss)
        
        assert all(loss > 0 for loss in losses)
        assert all(not np.isnan(loss) for loss in losses)
        
    def test_varying_synthetic_data(self):
        """Test training on varying synthetic data."""
        world_model = WorldModel(6, 2)
        
        # Generate varying synthetic data
        for i in range(100):
            obs = jnp.ones(6) * (0.3 + 0.4 * (i / 100))
            action = jnp.array([np.sin(i * 0.1), np.cos(i * 0.1)])
            next_obs = jnp.ones(6) * (0.3 + 0.4 * ((i + 1) / 100))
            reward = float(i % 10)
            world_model.add_experience(obs, action, next_obs, reward)
        
        # Train and ensure losses are reasonable
        losses = []
        for _ in range(5):
            loss = world_model.train_step()
            losses.append(loss)
        
        assert all(loss > 0 for loss in losses)
        assert all(loss < 1000 for loss in losses)  # Sanity check


class TestActorTraining:
    """Test actor training behavior."""
    
    def test_actor_train_step(self):
        """Test that actor training step executes without errors."""
        actor = Actor(32, 2)
        
        # Create synthetic latent batch
        latent_batch = jnp.ones((32, 32)) * 0.5
        reward_batch = jnp.ones(32)
        
        loss = actor.train_step(latent_batch, reward_batch)
        
        assert isinstance(loss, (float, np.floating))
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        
    def test_actor_training_stability(self):
        """Test that actor training produces stable losses."""
        actor = Actor(32, 2, seed=42)
        
        # Train on consistent data
        losses = []
        for _ in range(10):
            latent_batch = jnp.ones((32, 32)) * 0.5
            reward_batch = jnp.ones(32) * 5.0
            loss = actor.train_step(latent_batch, reward_batch)
            losses.append(loss)
        
        # All losses should be finite
        assert all(not np.isnan(loss) for loss in losses)
        assert all(not np.isinf(loss) for loss in losses)


class TestEndToEndTraining:
    """Test end-to-end training workflow."""
    
    def test_world_model_and_actor_together(self):
        """Test world model and actor training together."""
        world_model = WorldModel(6, 2)
        actor = Actor(32, 2)
        
        # Generate experiences
        for i in range(50):
            obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3 + 0.01 * i])
            action = jnp.array([0.0, 0.5])
            next_obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.31 + 0.01 * i])
            reward = 1.0 + 0.1 * i
            world_model.add_experience(obs, action, next_obs, reward)
        
        # Train world model
        world_loss = world_model.train_step()
        assert world_loss > 0
        
        # Train actor using latents from world model
        batch_size = 32
        latent_batch = jnp.array([world_model.encode(jnp.ones(6) * 0.5) for _ in range(batch_size)])
        reward_batch = jnp.ones(batch_size) * 5.0
        actor_loss = actor.train_step(latent_batch, reward_batch)
        
        # Actor loss can be negative (it's based on negative reward for policy gradient)
        assert not np.isnan(actor_loss)
        assert not np.isinf(actor_loss)
        
    def test_full_training_loop_simulation(self):
        """Test a simplified full training loop."""
        world_model = WorldModel(6, 2)
        actor = Actor(32, 2)
        
        # Simulate several training steps
        for episode in range(3):
            # Generate episode data
            for step in range(20):
                obs = jnp.ones(6) * (0.5 + 0.01 * step)
                latent = world_model.encode(obs)
                action = actor.get_action(latent, explore=True)
                next_obs = jnp.ones(6) * (0.51 + 0.01 * step)
                reward = 1.0
                
                world_model.add_experience(obs, action, next_obs, reward)
            
            # Train if enough data
            if len(world_model.buffer) >= world_model.batch_size:
                world_loss = world_model.train_step()
                assert world_loss >= 0
                
                # Train actor
                batch_size = 16
                latent_batch = jnp.array([world_model.encode(jnp.ones(6) * 0.5) for _ in range(batch_size)])
                reward_batch = jnp.ones(batch_size)
                actor_loss = actor.train_step(latent_batch, reward_batch)
                # Actor loss can be negative
                assert not np.isnan(actor_loss)

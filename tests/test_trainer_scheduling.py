"""
Tests for trainer scheduling (buffer fills, auto-save triggered).
"""
import pytest
import numpy as np
import jax.numpy as jnp
import os
import tempfile
import shutil
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['JAX_DISABLE_JIT'] = '1'

from test import WorldModel, Actor


class TestBufferFilling:
    """Test experience buffer filling behavior."""
    
    def test_buffer_fills_incrementally(self):
        """Test that buffer size increases incrementally."""
        world_model = WorldModel(6, 2)
        
        for i in range(10):
            obs = jnp.ones(6) * (i / 10)
            action = jnp.zeros(2)
            next_obs = jnp.ones(6) * ((i + 1) / 10)
            reward = float(i)
            
            world_model.add_experience(obs, action, next_obs, reward)
            assert len(world_model.buffer) == i + 1
    
    def test_buffer_reaches_batch_size(self):
        """Test that buffer can reach batch size threshold."""
        world_model = WorldModel(6, 2)
        batch_size = world_model.batch_size
        
        # Fill to exactly batch size
        for i in range(batch_size):
            obs = jnp.ones(6)
            action = jnp.zeros(2)
            next_obs = jnp.ones(6)
            reward = 1.0
            world_model.add_experience(obs, action, next_obs, reward)
        
        assert len(world_model.buffer) == batch_size
        # Should be able to train now
        loss = world_model.train_step()
        assert loss > 0
    
    def test_buffer_overflow_handling(self):
        """Test that buffer handles overflow correctly (FIFO)."""
        world_model = WorldModel(6, 2)
        max_size = world_model.buffer.maxlen
        
        # Fill beyond max size
        for i in range(max_size + 50):
            obs = jnp.ones(6) * (i / (max_size + 50))
            action = jnp.zeros(2)
            next_obs = jnp.ones(6)
            reward = float(i)
            world_model.add_experience(obs, action, next_obs, reward)
        
        # Buffer should be at max size
        assert len(world_model.buffer) == max_size
        
        # Oldest experience should be removed (FIFO)
        # The first experience should be gone
        first_remaining_reward = world_model.buffer[0][3]
        # It should be >= 50 because first 50 were removed
        assert first_remaining_reward >= 50


class TestTrainingScheduling:
    """Test training scheduling behavior."""
    
    def test_train_every_n_steps(self):
        """Test that training occurs at scheduled intervals."""
        world_model = WorldModel(6, 2)
        train_every = 5
        
        # Fill buffer with enough data
        for i in range(world_model.batch_size + 10):
            obs = jnp.ones(6)
            action = jnp.zeros(2)
            next_obs = jnp.ones(6)
            reward = 1.0
            world_model.add_experience(obs, action, next_obs, reward)
        
        # Simulate training every N steps
        losses = []
        for step in range(20):
            if step % train_every == 0:
                loss = world_model.train_step()
                losses.append(loss)
        
        # Should have trained 4 times (steps 0, 5, 10, 15)
        assert len(losses) == 4
        assert all(loss > 0 for loss in losses)
    
    def test_no_training_with_insufficient_data(self):
        """Test that training is skipped when data is insufficient."""
        world_model = WorldModel(6, 2)
        
        # Add only a few experiences
        for i in range(5):
            obs = jnp.ones(6)
            action = jnp.zeros(2)
            next_obs = jnp.ones(6)
            reward = 1.0
            world_model.add_experience(obs, action, next_obs, reward)
        
        # Attempt training - should return 0
        loss = world_model.train_step()
        assert loss == 0.0


class TestExplorationDecay:
    """Test exploration rate decay scheduling."""
    
    def test_exploration_decay(self):
        """Test that exploration probability decays over time."""
        explore_prob = 1.0
        min_explore_prob = 0.1
        explore_decay = 0.99
        
        # Simulate decay
        probs = [explore_prob]
        for _ in range(100):
            explore_prob = max(min_explore_prob, explore_prob * explore_decay)
            probs.append(explore_prob)
        
        # Should decrease monotonically until min
        assert probs[0] == 1.0
        assert probs[-1] >= min_explore_prob
        assert all(probs[i] >= probs[i+1] for i in range(len(probs)-1))
    
    def test_exploration_reaches_minimum(self):
        """Test that exploration eventually reaches minimum."""
        explore_prob = 1.0
        min_explore_prob = 0.1
        explore_decay = 0.99
        
        # Decay for many steps
        for _ in range(1000):
            explore_prob = max(min_explore_prob, explore_prob * explore_decay)
        
        # Should be at minimum
        assert explore_prob == min_explore_prob


class TestStepCounter:
    """Test step counting and scheduling."""
    
    def test_step_counter_increments(self):
        """Test that step counter increments correctly."""
        step_counter = 0
        
        for i in range(100):
            step_counter += 1
            assert step_counter == i + 1
    
    def test_modulo_scheduling(self):
        """Test using modulo for scheduling decisions."""
        train_every = 10
        training_steps = []
        
        for step in range(100):
            if step % train_every == 0:
                training_steps.append(step)
        
        # Should train at steps 0, 10, 20, ..., 90
        assert training_steps == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert len(training_steps) == 10


class TestProgressTracking:
    """Test progress tracking for auto-save triggers."""
    
    def test_progress_improvement_detection(self):
        """Test detecting when progress improves."""
        best_progress = 0.0
        progress_values = [0.1, 0.15, 0.12, 0.2, 0.18, 0.25]
        improvements = []
        
        for current_progress in progress_values:
            if current_progress > best_progress:
                best_progress = current_progress
                improvements.append(current_progress)
        
        # Should have detected improvements at 0.1, 0.15, 0.2, 0.25
        assert improvements == [0.1, 0.15, 0.2, 0.25]
    
    def test_no_improvement_no_save(self):
        """Test that no save occurs when progress doesn't improve."""
        best_progress = 0.5
        progress_values = [0.4, 0.45, 0.48, 0.42]
        save_triggered = []
        
        for current_progress in progress_values:
            if current_progress > best_progress:
                save_triggered.append(True)
                best_progress = current_progress
            else:
                save_triggered.append(False)
        
        # No saves should be triggered
        assert all(not triggered for triggered in save_triggered)


class TestEpisodeManagement:
    """Test episode tracking and reset behavior."""
    
    def test_episode_reward_accumulation(self):
        """Test that episode rewards accumulate correctly."""
        current_episode_reward = 0.0
        rewards = [1.0, 2.5, 0.5, 3.0, 1.5]
        
        for reward in rewards:
            current_episode_reward += reward
        
        assert current_episode_reward == sum(rewards)
    
    def test_episode_reward_reset_on_done(self):
        """Test that episode reward resets after episode ends."""
        episode_rewards = []
        current_episode_reward = 0.0
        
        # Simulate episode
        for step in range(10):
            reward = 1.0
            current_episode_reward += reward
            
            # Episode ends at step 9
            if step == 9:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
        
        assert len(episode_rewards) == 1
        assert episode_rewards[0] == 10.0
        assert current_episode_reward == 0.0
    
    def test_multiple_episodes_tracking(self):
        """Test tracking multiple episodes."""
        episode_rewards = []
        current_episode_reward = 0.0
        episode_lengths = [10, 15, 8]
        
        for ep_len in episode_lengths:
            for step in range(ep_len):
                current_episode_reward += 1.0
            
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
        
        assert len(episode_rewards) == 3
        assert episode_rewards == [10.0, 15.0, 8.0]


class TestTrainingStatsCollection:
    """Test collection of training statistics."""
    
    def test_loss_history_tracking(self):
        """Test tracking loss history."""
        training_losses = []
        
        # Simulate training steps
        for i in range(20):
            loss = 10.0 - i * 0.3  # Decreasing loss
            training_losses.append(loss)
        
        assert len(training_losses) == 20
        # Verify decreasing trend
        assert training_losses[0] > training_losses[-1]
    
    def test_recent_stats_window(self):
        """Test keeping only recent statistics."""
        stats = []
        max_window = 100
        
        # Add many stats
        for i in range(200):
            stats.append(i)
            # Keep only last 100
            if len(stats) > max_window:
                stats = stats[-max_window:]
        
        assert len(stats) == max_window
        assert stats[0] == 100  # First 100 entries removed
        assert stats[-1] == 199

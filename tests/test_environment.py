"""
Tests for environment dynamics (reset/step produce expected shapes).
"""
import pytest
import numpy as np
import jax.numpy as jnp
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['JAX_DISABLE_JIT'] = '1'

from test import Car, Track, WorldModel, Actor


class TestCarDynamics:
    """Test car physics and sensor behavior."""
    
    def test_car_initialization(self):
        """Test that car initializes with correct state shape."""
        car = Car(400, 300)
        assert car.state.shape == (4,)
        assert len(car.sensor_readings) == 5
        assert car.max_speed == 5.0
        
    def test_car_reset(self):
        """Test that car resets to initial position."""
        car = Car(400, 300)
        # Move the car
        car.state = jnp.array([500, 400, 3.0, 45.0])
        # Reset
        car.reset()
        assert car.state[0] == 400
        assert car.state[1] == 300
        assert car.state[2] == 0.0
        assert car.state[3] == 0.0
        
    def test_car_update_returns_bool(self):
        """Test that car update returns boolean (on_track status)."""
        car = Car(400, 300)
        track = Track()
        action = [0.0, 0.5]  # straight with throttle
        result = car.update(action, track.track_points, track.obstacles)
        assert isinstance(result, (bool, np.bool_))


class TestTrack:
    """Test track generation and properties."""
    
    def test_track_generation(self):
        """Test that track generates correct number of points."""
        track = Track()
        assert len(track.track_points) > 0
        # Track should be closed (first and last point same)
        assert track.track_points[0] == track.track_points[-1]
        
    def test_track_obstacles(self):
        """Test that obstacles are generated."""
        track = Track()
        assert len(track.obstacles) > 0
        # Each obstacle should be a tuple of coordinates
        for obs in track.obstacles:
            assert len(obs) == 2
            
    def test_deterministic_track(self):
        """Test that same seed produces same track."""
        track1 = Track(seed=42)
        track2 = Track(seed=42)
        assert track1.track_points == track2.track_points
        assert track1.obstacles == track2.obstacles


class TestEnvironmentShapes:
    """Test that environment produces expected shapes."""
    
    def test_observation_shape(self):
        """Test observation has correct shape (5 sensors + 1 speed)."""
        car = Car(400, 300)
        track = Track()
        car.update_sensors(track.track_points, track.obstacles)
        
        # Normalize sensor readings
        norm_sensors = np.array(car.sensor_readings) / car.sensor_length
        speed = float(car.state[2]) / car.max_speed
        obs = np.append(norm_sensors, speed)
        
        assert obs.shape == (6,)
        assert all(0 <= x <= 1 for x in obs)
        
    def test_action_shape(self):
        """Test that actions have correct shape and range."""
        action = jnp.array([0.5, 0.8])
        assert action.shape == (2,)
        # Actions should be in [-1, 1] range
        assert -1 <= action[0] <= 1
        assert -1 <= action[1] <= 1


class TestWorldModelIntegration:
    """Test world model integration with environment."""
    
    def test_world_model_initialization(self):
        """Test world model initializes with correct dimensions."""
        obs_size = 6  # 5 sensors + speed
        action_size = 2  # steering, throttle
        world_model = WorldModel(obs_size, action_size)
        
        assert world_model.obs_size == obs_size
        assert world_model.action_size == action_size
        assert world_model.latent_size == 32
        
    def test_encode_observation(self):
        """Test encoding observation to latent space."""
        world_model = WorldModel(6, 2)
        obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3])
        latent = world_model.encode(obs)
        
        assert latent.shape == (32,)
        assert not jnp.any(jnp.isnan(latent))
        
    def test_predict_next_state(self):
        """Test predicting next latent state and reward."""
        world_model = WorldModel(6, 2)
        latent = jnp.zeros(32)
        action = jnp.array([0.0, 0.5])
        
        next_latent, predicted_reward = world_model.predict_next(latent, action)
        
        assert next_latent.shape == (32,)
        assert isinstance(predicted_reward, (float, np.floating))
        assert not jnp.any(jnp.isnan(next_latent))
        
    def test_decode_latent(self):
        """Test decoding latent state back to observation."""
        world_model = WorldModel(6, 2)
        latent = jnp.zeros(32)
        decoded_obs = world_model.decode(latent)
        
        assert decoded_obs.shape == (6,)
        assert not jnp.any(jnp.isnan(decoded_obs))


class TestActorIntegration:
    """Test actor policy integration."""
    
    def test_actor_initialization(self):
        """Test actor initializes correctly."""
        latent_size = 32
        action_size = 2
        actor = Actor(latent_size, action_size)
        
        assert actor.latent_size == latent_size
        assert actor.action_size == action_size
        
    def test_get_action(self):
        """Test getting action from latent state."""
        actor = Actor(32, 2)
        latent = jnp.zeros(32)
        action = actor.get_action(latent, explore=False)
        
        assert action.shape == (2,)
        # Actions should be in [-1, 1] range (tanh output)
        assert jnp.all(action >= -1.0)
        assert jnp.all(action <= 1.0)
        
    def test_exploration_adds_noise(self):
        """Test that exploration mode adds variability."""
        actor = Actor(32, 2, seed=42)
        latent = jnp.zeros(32)
        
        # Get multiple actions without exploration
        actions_no_explore = [actor.get_action(latent, explore=False) for _ in range(5)]
        
        # All should be identical
        for i in range(1, len(actions_no_explore)):
            assert jnp.allclose(actions_no_explore[0], actions_no_explore[i])
        
        # Actions with exploration should vary (due to random key splitting)
        actions_explore = [actor.get_action(latent, explore=True) for _ in range(5)]
        # At least some should be different
        differences = sum(not jnp.allclose(actions_explore[0], a) for a in actions_explore[1:])
        assert differences > 0

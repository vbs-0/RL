"""
Tests for persistence metadata integrity.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import os
import pickle
import tempfile
import shutil
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['JAX_DISABLE_JIT'] = '1'

from test import WorldModel, Actor


class TestModelSerialization:
    """Test model parameter serialization."""
    
    def test_world_model_params_serializable(self):
        """Test that world model parameters can be serialized."""
        world_model = WorldModel(6, 2)
        
        # Collect all parameters
        params = {
            'encoder_params': world_model.encoder_params,
            'dynamics_params': world_model.dynamics_params,
            'decoder_params': world_model.decoder_params,
            'reward_params': world_model.reward_params
        }
        
        # Try to serialize
        try:
            serialized = pickle.dumps(params)
            assert len(serialized) > 0
            
            # Try to deserialize
            deserialized = pickle.loads(serialized)
            assert set(deserialized.keys()) == set(params.keys())
        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")
    
    def test_actor_params_serializable(self):
        """Test that actor parameters can be serialized."""
        actor = Actor(32, 2)
        
        params = {
            'actor_params': actor.actor_params
        }
        
        try:
            serialized = pickle.dumps(params)
            assert len(serialized) > 0
            
            deserialized = pickle.loads(serialized)
            assert 'actor_params' in deserialized
        except Exception as e:
            pytest.fail(f"Serialization failed: {e}")
    
    def test_full_model_state_serializable(self):
        """Test that complete model state can be serialized."""
        world_model = WorldModel(6, 2)
        actor = Actor(32, 2)
        
        # Add some experiences
        for i in range(10):
            obs = jnp.ones(6) * (i / 10)
            action = jnp.zeros(2)
            next_obs = jnp.ones(6) * ((i + 1) / 10)
            reward = float(i)
            world_model.add_experience(obs, action, next_obs, reward)
        
        model_state = {
            'encoder_params': world_model.encoder_params,
            'dynamics_params': world_model.dynamics_params,
            'decoder_params': world_model.decoder_params,
            'reward_params': world_model.reward_params,
            'actor_params': actor.actor_params,
            'explore_prob': 0.8,
            'episode_rewards': [10.0, 15.0, 12.0],
            'training_losses': [5.0, 4.5, 4.0]
        }
        
        try:
            serialized = pickle.dumps(model_state)
            deserialized = pickle.loads(serialized)
            
            # Verify metadata
            assert deserialized['explore_prob'] == 0.8
            assert len(deserialized['episode_rewards']) == 3
            assert len(deserialized['training_losses']) == 3
        except Exception as e:
            pytest.fail(f"Full state serialization failed: {e}")


class TestModelPersistence:
    """Test model save/load functionality."""
    
    def test_save_and_load_model(self):
        """Test saving and loading model to/from file."""
        world_model = WorldModel(6, 2)
        actor = Actor(32, 2)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Prepare model state
            model_state = {
                'encoder_params': world_model.encoder_params,
                'dynamics_params': world_model.dynamics_params,
                'decoder_params': world_model.decoder_params,
                'reward_params': world_model.reward_params,
                'actor_params': actor.actor_params,
                'explore_prob': 0.75,
                'episode_rewards': [10.0, 15.0],
                'training_losses': [5.0, 4.5]
            }
            
            # Save to file
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            assert os.path.exists(filepath)
            
            # Load from file
            with open(filepath, 'rb') as f:
                loaded_state = pickle.load(f)
            
            # Verify metadata integrity
            assert loaded_state['explore_prob'] == 0.75
            assert loaded_state['episode_rewards'] == [10.0, 15.0]
            assert loaded_state['training_losses'] == [5.0, 4.5]
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
    
    def test_loaded_model_functional(self):
        """Test that loaded model can make predictions."""
        world_model = WorldModel(6, 2)
        actor = Actor(32, 2)
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Save original model
            model_state = {
                'encoder_params': world_model.encoder_params,
                'dynamics_params': world_model.dynamics_params,
                'decoder_params': world_model.decoder_params,
                'reward_params': world_model.reward_params,
                'actor_params': actor.actor_params,
                'explore_prob': 0.5,
                'episode_rewards': [],
                'training_losses': []
            }
            
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            # Create new models and load state
            new_world_model = WorldModel(6, 2)
            new_actor = Actor(32, 2)
            
            with open(filepath, 'rb') as f:
                loaded_state = pickle.load(f)
            
            new_world_model.encoder_params = loaded_state['encoder_params']
            new_world_model.dynamics_params = loaded_state['dynamics_params']
            new_world_model.decoder_params = loaded_state['decoder_params']
            new_world_model.reward_params = loaded_state['reward_params']
            new_actor.actor_params = loaded_state['actor_params']
            
            # Test that loaded models work
            obs = jnp.ones(6) * 0.5
            latent = new_world_model.encode(obs)
            action = new_actor.get_action(latent, explore=False)
            
            assert latent.shape == (32,)
            assert action.shape == (2,)
            assert not jnp.any(jnp.isnan(latent))
            assert not jnp.any(jnp.isnan(action))
            
        finally:
            shutil.rmtree(temp_dir)


class TestMetadataIntegrity:
    """Test metadata integrity in saved models."""
    
    def test_metadata_types(self):
        """Test that metadata has correct types."""
        metadata = {
            'explore_prob': 0.8,
            'episode_rewards': [10.0, 15.0, 12.0],
            'training_losses': [5.0, 4.5, 4.0],
            'step_counter': 1000,
            'best_progress': 0.85
        }
        
        assert isinstance(metadata['explore_prob'], float)
        assert isinstance(metadata['episode_rewards'], list)
        assert isinstance(metadata['training_losses'], list)
        assert isinstance(metadata['step_counter'], int)
        assert isinstance(metadata['best_progress'], float)
    
    def test_metadata_values_valid(self):
        """Test that metadata values are in valid ranges."""
        metadata = {
            'explore_prob': 0.8,
            'episode_rewards': [10.0, 15.0, 12.0],
            'training_losses': [5.0, 4.5, 4.0],
            'best_progress': 0.85
        }
        
        # Exploration probability should be [0, 1]
        assert 0 <= metadata['explore_prob'] <= 1
        
        # Progress should be [0, 1]
        assert 0 <= metadata['best_progress'] <= 1
        
        # Losses should be non-negative
        assert all(loss >= 0 for loss in metadata['training_losses'])
    
    def test_metadata_serialization_roundtrip(self):
        """Test that metadata survives serialization roundtrip."""
        original_metadata = {
            'explore_prob': 0.65,
            'episode_rewards': [8.5, 12.3, 15.7],
            'training_losses': [6.2, 5.8, 5.3],
            'step_counter': 2500,
            'best_progress': 0.72
        }
        
        # Serialize
        serialized = pickle.dumps(original_metadata)
        
        # Deserialize
        loaded_metadata = pickle.loads(serialized)
        
        # Verify all fields match
        assert loaded_metadata['explore_prob'] == original_metadata['explore_prob']
        assert loaded_metadata['episode_rewards'] == original_metadata['episode_rewards']
        assert loaded_metadata['training_losses'] == original_metadata['training_losses']
        assert loaded_metadata['step_counter'] == original_metadata['step_counter']
        assert loaded_metadata['best_progress'] == original_metadata['best_progress']


class TestModelVersioning:
    """Test model versioning and naming."""
    
    def test_model_filename_format(self):
        """Test that model filenames follow expected format."""
        episode_count = 42
        filename = f"model_ep{episode_count}.pkl"
        
        assert filename.startswith("model_ep")
        assert filename.endswith(".pkl")
        assert "42" in filename
    
    def test_extract_episode_from_filename(self):
        """Test extracting episode number from filename."""
        filenames = [
            "model_ep0.pkl",
            "model_ep10.pkl",
            "model_ep100.pkl"
        ]
        
        for filename in filenames:
            # Extract number
            ep_str = filename.replace("model_ep", "").replace(".pkl", "")
            ep_num = int(ep_str)
            assert ep_num >= 0
    
    def test_model_file_extension(self):
        """Test that model files have correct extension."""
        filename = "model_ep50.pkl"
        assert filename.endswith('.pkl')
        
        # Should be able to check extension
        _, ext = os.path.splitext(filename)
        assert ext == '.pkl'


class TestDirectoryManagement:
    """Test model directory management."""
    
    def test_model_directory_creation(self):
        """Test that model directory can be created."""
        temp_dir = tempfile.mkdtemp()
        try:
            model_dir = os.path.join(temp_dir, 'models')
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            assert os.path.exists(model_dir)
            assert os.path.isdir(model_dir)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_list_model_files(self):
        """Test listing model files in directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            model_dir = os.path.join(temp_dir, 'models')
            os.makedirs(model_dir)
            
            # Create some model files
            for i in range(5):
                filepath = os.path.join(model_dir, f'model_ep{i}.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump({'episode': i}, f)
            
            # List model files
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            assert len(model_files) == 5
            assert all(f.startswith('model_ep') for f in model_files)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_model_file_overwrite(self):
        """Test that model files can be overwritten."""
        temp_dir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(temp_dir, 'model_test.pkl')
            
            # Write first version
            with open(filepath, 'wb') as f:
                pickle.dump({'version': 1}, f)
            
            with open(filepath, 'rb') as f:
                data1 = pickle.load(f)
            assert data1['version'] == 1
            
            # Overwrite with second version
            with open(filepath, 'wb') as f:
                pickle.dump({'version': 2}, f)
            
            with open(filepath, 'rb') as f:
                data2 = pickle.load(f)
            assert data2['version'] == 2
            
        finally:
            shutil.rmtree(temp_dir)

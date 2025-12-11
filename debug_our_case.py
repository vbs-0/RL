#!/usr/bin/env python3
"""Debug the actual JAX grad structure for our specific case."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random, grad
import haiku as hk
import optax
import numpy as np
from collections import deque

# Disable JIT
jax.config.update('jax_disable_jit', True)

def debug_our_case():
    """Debug the exact structure for our world model case."""
    
    print("Testing our world model case...")
    
    # Create the same setup as our WorldModel
    obs_size = 6
    action_size = 2
    latent_size = 32
    
    key = random.PRNGKey(42)
    
    # Define networks exactly like in WorldModel
    def encoder_fn(obs):
        net = hk.Sequential([
            hk.Linear(64), jax.nn.relu,
            hk.Linear(latent_size)
        ])
        return net(obs)
    
    def dynamics_fn(latent, action):
        net = hk.Sequential([
            hk.Linear(64), jax.nn.relu,
            hk.Linear(latent_size)
        ])
        x = jnp.concatenate([latent, action], axis=-1)
        return net(x)
    
    def decoder_fn(latent):
        net = hk.Sequential([
            hk.Linear(64), jax.nn.relu,
            hk.Linear(obs_size)
        ])
        return net(latent)
    
    def reward_fn(latent):
        net = hk.Sequential([
            hk.Linear(32), jax.nn.relu,
            hk.Linear(1)
        ])
        return net(latent)
    
    # Transform into pure functions
    encoder = hk.transform(encoder_fn)
    dynamics = hk.transform(dynamics_fn)
    decoder = hk.transform(decoder_fn)
    reward = hk.transform(reward_fn)
    
    # Initialize parameters
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_latent = jnp.zeros((1, latent_size))
    dummy_action = jnp.zeros((1, action_size))
    
    key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
    encoder_params = encoder.init(subkey1, dummy_obs)
    dynamics_params = dynamics.init(subkey2, dummy_latent, dummy_action)
    decoder_params = decoder.init(subkey3, dummy_latent)
    reward_params = reward.init(subkey4, dummy_latent)
    
    # Create dummy batch data
    batch_size = 4
    obs_batch = jnp.zeros((batch_size, obs_size))
    action_batch = jnp.zeros((batch_size, action_size))
    next_obs_batch = jnp.zeros((batch_size, obs_size))
    reward_batch = jnp.zeros((batch_size,))
    
    # Define the same loss function
    def compute_loss(encoder_params, dynamics_params, decoder_params, reward_params):
        # Get latent representations
        latent_batch = encoder.apply(encoder_params, None, obs_batch)
        
        # Predict next latent and reward
        next_latent_batch = dynamics.apply(dynamics_params, None, latent_batch, action_batch)
        predicted_reward = reward.apply(reward_params, None, latent_batch)
        
        # Decode to observations
        decoded_obs = decoder.apply(decoder_params, None, latent_batch)
        decoded_next_obs = decoder.apply(decoder_params, None, next_latent_batch)
        
        # Calculate losses
        reconstruction_loss = jnp.mean((decoded_obs - obs_batch) ** 2)
        next_reconstruction_loss = jnp.mean((decoded_next_obs - next_obs_batch) ** 2)
        reward_loss = jnp.mean((predicted_reward.squeeze() - reward_batch) ** 2)
        
        total_loss = reconstruction_loss + next_reconstruction_loss + reward_loss
        return total_loss, (reconstruction_loss, next_reconstruction_loss, reward_loss)
    
    # Test grad computation
    print("Computing gradients...")
    grads_result = grad(compute_loss, argnums=(0, 1, 2, 3), has_aux=True)(
        encoder_params, dynamics_params, decoder_params, reward_params
    )
    
    print(f"Result type: {type(grads_result)}")
    print(f"Result length: {len(grads_result)}")
    
    # Examine structure
    for i, item in enumerate(grads_result):
        print(f"Item {i}:")
        print(f"  Type: {type(item)}")
        if hasattr(item, '__len__') and not isinstance(item, (jax.Array, np.ndarray)):
            print(f"  Length: {len(item)}")
            if isinstance(item, tuple):
                for j, subitem in enumerate(item):
                    print(f"    Subitem {j}: type {type(subitem)}")
                    if j < 3:  # Only print first few to avoid too much output
                        try:
                            if hasattr(subitem, 'shape'):
                                print(f"      Shape: {subitem.shape}")
                        except:
                            pass

if __name__ == '__main__':
    debug_our_case()
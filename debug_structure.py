#!/usr/bin/env python3
"""Debug the actual JAX grad structure and fix the order."""

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

def debug_grad_structure():
    """Debug the JAX grad structure and parameter compatibility."""
    
    print("Testing JAX grad structure with parameter matching...")
    
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
    
    print(f"Result structure:")
    print(f"  Type: {type(grads_result)}")
    print(f"  Length: {len(grads_result)}")
    
    # Check what we got
    first_item = grads_result[0]
    second_item = grads_result[1]
    
    print(f"First item type: {type(first_item)}")
    print(f"Second item type: {type(second_item)}")
    
    if hasattr(first_item, '__len__') and not isinstance(first_item, (jax.Array, np.ndarray)):
        print(f"First item length: {len(first_item)}")
    if hasattr(second_item, '__len__') and not isinstance(second_item, (jax.Array, np.ndarray)):
        print(f"Second item length: {len(second_item)}")
    
    # Test both interpretations
    print("\nTesting interpretation 1: (losses, gradients)")
    try:
        loss_aux, gradients = grads_result
        print(f"  Loss aux type: {type(loss_aux)}")
        print(f"  Gradients type: {type(gradients)}")
        if hasattr(gradients, '__len__'):
            print(f"  Gradients length: {len(gradients)}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    print("\nTesting interpretation 2: (gradients, losses)")
    try:
        gradients, loss_aux = grads_result
        print(f"  Gradients type: {type(gradients)}")
        print(f"  Loss aux type: {type(loss_aux)}")
        if hasattr(gradients, '__len__'):
            print(f"  Gradients length: {len(gradients)}")
        if hasattr(loss_aux, '__len__'):
            print(f"  Loss aux length: {len(loss_aux)}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test gradient structure compatibility
    print("\nTesting gradient structure...")
    try:
        gradients, loss_aux = grads_result
        enc_grad, dyn_grad, dec_grad, rew_grad = gradients
        print(f"  ✓ Can unpack 4 gradients")
        
        # Test tree structure compatibility
        print(f"  Encoder params keys: {list(encoder_params.keys())}")
        print(f"  Encoder grad keys: {list(enc_grad.keys())}")
        
        # Test optimizer compatibility
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(encoder_params)
        updates, new_state = optimizer.update(enc_grad, opt_state)
        print(f"  ✓ Optimizer update works")
        
    except Exception as e:
        print(f"  ✗ Gradient compatibility failed: {e}")

if __name__ == '__main__':
    debug_grad_structure()
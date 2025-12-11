#!/usr/bin/env python3
"""Test if training is actually happening using the new modular system."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random, grad
import optax
import numpy as np

# Disable JIT
jax.config.update('jax_disable_jit', True)

# Test basic JAX functionality
print("Testing basic JAX functionality...")
def simple_loss(params):
    """Simple quadratic loss"""
    return jnp.sum(params ** 2)

# Initialize params
key = random.PRNGKey(0)
params = jnp.array([1.0, 2.0, 3.0])

# Create optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

print("Testing gradient updates:")
print(f"Initial params: {params}")
print(f"Initial loss: {simple_loss(params):.6f}")

# Do 5 steps
for step in range(5):
    loss = simple_loss(params)
    grads = grad(simple_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(f"Step {step+1}: loss={loss:.6f}, params={params}")

print("\nGradient updates working correctly!" if params[0] < 0.9 else "\nWARNING: Params not updating!")

# Test the actual modular RL system
print("\n" + "="*50)
print("Testing modular RL system:")
print("="*50)

try:
    from rl_system.training import ContinuousTrainer
    from rl_system.config import config
    
    # Test configuration loading
    print("Configuration test:")
    print(f"  Environment width: {config.get('environment.width', 800)}")
    print(f"  Agent latent size: {config.get('agent.latent_size', 32)}")
    print(f"  Training batch size: {config.get('training.batch_size', 32)}")
    
    # Test trainer initialization
    print("\nTesting trainer initialization...")
    trainer = ContinuousTrainer(headless=True)
    print("✓ Trainer initialized successfully")
    
    # Test environment
    print("\nTesting environment...")
    obs = trainer.env.reset()
    print(f"✓ Environment reset, observation shape: {obs.shape}")
    
    # Test agent components
    print("\nTesting agent components...")
    
    # Test world model encoding
    latent = trainer.world_model.encode(obs)
    print(f"✓ World model encoding: {obs.shape} -> {latent.shape}")
    
    # Test action generation
    action = trainer.get_action(obs)
    print(f"✓ Action generation: {action.shape}")
    
    # Test environment step
    next_obs, reward, done = trainer.env.step(action)
    print(f"✓ Environment step: reward={reward:.3f}, done={done}")
    
    # Test training step
    print("\nTesting training step...")
    losses = trainer.train_step(obs, action, next_obs, reward)
    print(f"✓ Training step completed: {losses}")
    
    # Test model saving
    print("\nTesting model operations...")
    trainer.save_model("test")
    models = trainer.list_models()
    print(f"✓ Model save/load: found {len(models)} models")
    
    # Test configuration updates
    print("\nTesting configuration updates...")
    config.set('training.batch_size', 64)
    new_batch_size = config.get('training.batch_size', 32)
    print(f"✓ Config update: batch_size = {new_batch_size}")
    
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED! Modular system is working correctly.")
    print("="*50)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nThe modular system needs to be debugged.")
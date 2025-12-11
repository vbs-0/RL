#!/usr/bin/env python3
"""Test the exact optimizer issue."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random, grad
import haiku as hk
import optax

# Disable JIT
jax.config.update('jax_disable_jit', True)

def test_optimizer_issue():
    """Test the exact optimizer issue step by step."""
    
    print("Testing optimizer step by step...")
    
    # Create simple network like in our system
    def test_fn(x):
        return hk.Linear(4)(x)
    
    # Transform and initialize
    transform = hk.transform(test_fn)
    key = random.PRNGKey(42)
    dummy_input = jnp.zeros((2, 3))
    params = transform.init(key, dummy_input)
    
    print(f"Params type: {type(params)}")
    print(f"Params keys: {list(params.keys())}")
    
    # Create optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    print(f"Opt state type: {type(opt_state)}")
    
    # Create dummy loss function
    def loss_fn(params):
        dummy = jnp.ones((1, 3))
        output = transform.apply(params, None, dummy)
        return jnp.mean(output ** 2)
    
    # Get gradients
    grads = grad(loss_fn)(params)
    
    print(f"Grads type: {type(grads)}")
    print(f"Grads keys: {list(grads.keys())}")
    
    # Try optimizer update
    try:
        print("Trying optimizer update...")
        updates, new_opt_state = optimizer.update(grads, opt_state)
        print("✓ Optimizer update successful")
        
        # Try applying updates
        new_params = optax.apply_updates(params, updates)
        print("✓ Apply updates successful")
        
    except Exception as e:
        print(f"✗ Optimizer update failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with more complex structure
    print("\nTesting with multi-layer network...")
    
    def multi_fn(x):
        return hk.Sequential([
            hk.Linear(32), jax.nn.relu,
            hk.Linear(16), jax.nn.relu,
            hk.Linear(4)
        ])(x)
    
    multi_transform = hk.transform(multi_fn)
    multi_params = multi_transform.init(key, dummy_input)
    
    print(f"Multi params structure:")
    for k, v in multi_params.items():
        print(f"  {k}: {type(v)}")
        if hasattr(v, 'keys'):
            for k2, v2 in v.items():
                print(f"    {k2}: {type(v2)}, shape {getattr(v2, 'shape', 'N/A')}")
    
    multi_opt_state = optimizer.init(multi_params)
    multi_grads = grad(lambda p: loss_fn(p))(multi_params)  # Using same loss for simplicity
    
    try:
        print("Trying multi-layer optimizer update...")
        updates, new_opt_state = optimizer.update(multi_grads, multi_opt_state)
        print("✓ Multi-layer optimizer update successful")
    except Exception as e:
        print(f"✗ Multi-layer optimizer update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_optimizer_issue()
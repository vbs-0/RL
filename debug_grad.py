#!/usr/bin/env python3
"""Debug JAX gradient computation."""

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

def debug_grad_computation():
    """Debug the gradient computation issue."""
    
    print("Testing JAX gradient computation...")
    
    # Create simple test networks
    def test_fn(x):
        return hk.Linear(4)(x)
    
    # Transform and initialize
    transform = hk.transform(test_fn)
    key = random.PRNGKey(42)
    dummy_input = jnp.zeros((2, 3))
    params1 = transform.init(key, dummy_input)
    params2 = transform.init(key, dummy_input)
    
    def simple_loss(p1, p2):
        # Simple loss that combines both params
        dummy = jnp.ones((1, 3))
        out1 = transform.apply(p1, None, dummy)
        out2 = transform.apply(p2, None, dummy)
        return jnp.mean((out1 - out2) ** 2), (jnp.mean(out1), jnp.mean(out2))
    
    # Test grad computation
    print("Testing grad with has_aux=True...")
    result = grad(simple_loss, argnums=(0, 1), has_aux=True)(params1, params2)
    
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
    
    if isinstance(result, tuple) and len(result) == 2:
        (loss_val, loss_aux), gradients = result
        print(f"Loss value: {loss_val}")
        print(f"Loss auxiliary: {loss_aux}")
        print(f"Gradients type: {type(gradients)}")
        if hasattr(gradients, '__len__'):
            print(f"Gradients length: {len(gradients)}")
    else:
        print(f"Unexpected result structure: {result}")
    
    # Test the actual multi-parameter case
    print("\nTesting actual multi-parameter grad...")
    
    def multi_loss(p1, p2, p3, p4):
        # Return a tuple for loss components
        return 1.0, (0.5, 0.3, 0.2)
    
    # Initialize some dummy params
    p1 = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array([0.1])}
    p2 = {'w': jnp.array([2.0, 3.0]), 'b': jnp.array([0.2])}
    p3 = {'w': jnp.array([3.0, 4.0]), 'b': jnp.array([0.3])}
    p4 = {'w': jnp.array([4.0, 5.0]), 'b': jnp.array([0.4])}
    
    try:
        result = grad(multi_loss, argnums=(0, 1, 2, 3), has_aux=True)(p1, p2, p3, p4)
        print(f"Multi-param result type: {type(result)}")
        
        if isinstance(result, tuple):
            print(f"Multi-param result length: {len(result)}")
            for i, item in enumerate(result):
                print(f"  Item {i}: type {type(item)}, length {len(item) if hasattr(item, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"Multi-param grad error: {e}")

if __name__ == '__main__':
    debug_grad_computation()
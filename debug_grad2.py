#!/usr/bin/env python3
"""Debug the exact JAX gradient structure."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random, grad

# Disable JIT
jax.config.update('jax_disable_jit', True)

def debug_exact_structure():
    """Debug the exact structure returned by JAX grad."""
    
    print("Testing exact JAX grad structure...")
    
    # Create dummy parameters
    p1 = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array([0.1])}
    p2 = {'w': jnp.array([2.0, 3.0]), 'b': jnp.array([0.2])}
    p3 = {'w': jnp.array([3.0, 4.0]), 'b': jnp.array([0.3])}
    p4 = {'w': jnp.array([4.0, 5.0]), 'b': jnp.array([0.4])}
    
    def test_loss(p1, p2, p3, p4):
        # Return loss and auxiliary data
        loss = p1['w'][0] + p2['w'][0] + p3['w'][0] + p4['w'][0]
        aux = (loss * 2, loss * 3)  # Some auxiliary data
        return loss, aux
    
    result = grad(test_loss, argnums=(0, 1, 2, 3), has_aux=True)(p1, p2, p3, p4)
    
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")
    
    # Examine each item
    for i, item in enumerate(result):
        print(f"Item {i}:")
        print(f"  Type: {type(item)}")
        if hasattr(item, '__len__'):
            print(f"  Length: {len(item)}")
            if isinstance(item, tuple):
                for j, subitem in enumerate(item):
                    print(f"    Subitem {j}: type {type(subitem)}")
    
    # Try the unpacking
    try:
        loss_aux, grads_tuple = result
        print(f"\nUnpacked successfully:")
        print(f"Loss/aux type: {type(loss_aux)}")
        print(f"Grads type: {type(grads_tuple)}")
        print(f"Grads length: {len(grads_tuple)}")
    except Exception as e:
        print(f"\nUnpacking failed: {e}")

if __name__ == '__main__':
    debug_exact_structure()
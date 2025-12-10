#!/usr/bin/env python3
"""Test if training is actually happening"""

import jax
import jax.numpy as jnp
from jax import random, grad
import optax
import numpy as np

# Disable JIT
jax.config.update('jax_disable_jit', True)

# Simple test of gradient updates
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

# Now test our actual network training
print("\n" + "="*50)
print("Testing actual network training:")
print("="*50)

import haiku as hk

# Simple network
def network_fn(x):
    net = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1)
    ])
    return net(x)

# Initialize
transform = hk.transform(network_fn)
key, subkey = random.split(random.PRNGKey(42))
dummy_input = jnp.zeros((1, 10))
net_params = transform.init(subkey, dummy_input)

# Create optimizer
net_optimizer = optax.adam(learning_rate=1e-3)
net_opt_state = net_optimizer.init(net_params)

# Create dummy data
batch_x = random.normal(key, (8, 10))
batch_y = jnp.sum(batch_x ** 2, axis=1, keepdims=True)

# Define loss
def compute_loss(params):
    y_pred = transform.apply(params, None, batch_x)
    loss = jnp.mean((y_pred - batch_y) ** 2)
    return loss

print(f"Initial loss: {compute_loss(net_params):.6f}")

# Train for 5 steps
for step in range(5):
    loss = compute_loss(net_params)
    grads = grad(compute_loss)(net_params)
    updates, net_opt_state = net_optimizer.update(grads, net_opt_state)
    net_params = optax.apply_updates(net_params, updates)
    print(f"Step {step+1}: loss={loss:.6f}")

final_loss = compute_loss(net_params)
print(f"\nNetwork training working: {final_loss < 1.0}")

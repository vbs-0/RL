# Architecture Guide

This document provides a comprehensive overview of the system architecture, component interactions, and data flow patterns in the RL Car Racing project.

## Table of Contents
1. [System Overview](#system-overview)
2. [Package Layout](#package-layout)
3. [Core Components](#core-components)
4. [Trainer Lifecycle](#trainer-lifecycle)
5. [Threading Model](#threading-model)
6. [Data Flow](#data-flow)
7. [Neural Network Architecture](#neural-network-architecture)
8. [Model Persistence](#model-persistence)

## System Overview

The project is a web-based reinforcement learning training platform consisting of:

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser Client                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ MJPEG Stream │  │   Controls   │  │ Training Charts  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────┐
│                      Flask Web Server                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Endpoints (start/stop/stats/control)        │   │
│  └─────────────────────┬────────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Game Loop Thread                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Pygame     │  │   RL Agent   │  │  Frame Renderer  │  │
│  │  Simulation  │◄─┤ (JAX/Haiku)  │  │   (Headless)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key Technologies**:
- **Flask**: HTTP server for API and video streaming
- **Pygame**: Physics simulation and rendering
- **JAX**: Hardware-accelerated numerical computation
- **Haiku**: Neural network library on top of JAX
- **Optax**: Gradient-based optimization (used in test.py)
- **Threading**: Concurrent game loop and web server

## Package Layout

```
project/
├── app.py                    # Flask web server + headless game integration
├── game.py                   # Standalone RL car game with Pygame display
├── test.py                   # Enhanced version with Optax optimizers
├── car_game.py               # Simpler demo with manual controls
├── test_training.py          # Training sanity check script
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Web dashboard UI
├── models/                   # Persisted model checkpoints (*.pkl)
└── docs/                    # Documentation (this file)
```

### File Responsibilities

**app.py** (Production Web Server)
- Main entry point for remote training
- Wraps game in headless Pygame surface
- Streams MJPEG video feed
- Exposes REST API for control
- Manages game loop in background thread
- Handles model save/load operations

**test.py** (Advanced Training Module)
- Complete implementation with Optax optimizers
- Used as the base module imported by app.py
- Includes full Car, Track, WorldModel, Actor classes
- Production-ready training loop

**game.py** (Standalone Demo)
- Can run independently for local testing
- Displays Pygame window directly
- Same core logic as test.py
- Useful for debugging without web interface

**car_game.py** (Simple Demo)
- Minimal example for understanding basics
- Manual keyboard control only
- No RL components

**test_training.py** (Sanity Check)
- Validates JAX/Haiku setup
- Quick smoke test for training loop

## Core Components

### 1. Car Class

**Location**: `test.py`, `game.py`

**Responsibilities**:
- Physics simulation (position, velocity, angle)
- Distance sensor ray casting
- Collision detection with track boundaries
- Track progress calculation

**State Vector**: `[x, y, speed, angle]`
- `x, y`: Position in pixels (0-800, 0-600)
- `speed`: Current velocity (0 to 5.0)
- `angle`: Heading in degrees (0-360)

**Key Methods**:
```python
update(action, track_points, obstacles) -> bool
    # Updates physics based on [steering, throttle] action
    # Returns True if still on track, False if crashed

update_sensors(track_points, obstacles) -> None
    # Casts 5 rays to measure distances to boundaries/obstacles
    # Updates self.sensor_readings array

is_on_track(track_points) -> bool
    # Checks if car center is within track_width of any segment

calculate_track_progress(track_points) -> float
    # Returns progress from 0.0 (start) to 1.0 (complete lap)
```

### 2. Track Class

**Location**: `test.py`, `game.py`

**Responsibilities**:
- Procedural track generation using Perlin noise
- Obstacle placement
- Rendering track visualization

**Generation Algorithm**:
1. Create 100 points in a circle around screen center
2. Apply Perlin noise to vary radius organically
3. Connect points to form closed loop
4. Place 5 obstacles using noise-based positioning

**Customization**:
- `seed`: Controls random generation (same seed = same track)
- `track_width`: 40 pixels (hardcoded)
- `checkpoint_interval`: 0.1 (10% progress increments)

### 3. WorldModel Class

**Location**: `test.py`, `game.py`

**Responsibilities**:
- Encode observations to latent representations
- Predict next states given current state + action
- Predict rewards from states
- Decode latent states back to observations
- Manage experience replay buffer
- Train on sampled experiences

**Architecture**:
```
Encoder:     [6 obs] → [64 ReLU] → [32 latent]
Dynamics:    [32 latent + 2 action] → [64 ReLU] → [32 next_latent]
Reward:      [32 latent] → [32 ReLU] → [1 reward]
Decoder:     [32 latent] → [64 ReLU] → [6 reconstructed_obs]
```

**Training Process**:
1. Sample batch from replay buffer (32 experiences)
2. Encode observations to latent space
3. Predict next latents using dynamics model
4. Predict rewards
5. Decode latents to reconstruct observations
6. Calculate losses:
   - Reconstruction loss: MSE(decoded_obs, real_obs)
   - Next state loss: MSE(decoded_next_obs, real_next_obs)
   - Reward loss: MSE(predicted_reward, actual_reward)
7. Total loss = sum of all losses
   *(Note: In current implementation, parameters aren't updated, only loss is computed)*

**Buffer Management**:
- Type: `deque` with `maxlen=10000`
- Stores tuples: `(obs, action, next_obs, reward)`
- FIFO replacement when full

### 4. Actor Class

**Location**: `test.py`, `game.py`

**Responsibilities**:
- Map latent states to actions
- Add exploration noise when needed
- Provide deterministic policy for evaluation

**Architecture**:
```
Actor: [32 latent] → [64 ReLU] → [32 ReLU] → [2 tanh]
```

**Output**:
- 2D action vector: `[steering, throttle]`
- Both in range [-1.0, 1.0] due to `tanh` activation

**Exploration**:
```python
if explore:
    noise = random.normal(key, shape=action.shape) * 0.2
    action = clip(action + noise, -1.0, 1.0)
```

### 5. FlaskRLCarEnv Class

**Location**: `app.py`

**Responsibilities**:
- Wrapper around Car + Track for web integration
- Headless rendering to PIL Image → JPEG → base64
- Model save/load with pickle
- Statistics tracking for dashboard

**Key Differences from RLCarEnv**:
- No Pygame display window (uses SDL dummy driver)
- Renders to Surface instead of screen
- Automatic model saving on progress improvement
- Global game_state dict for thread communication

## Trainer Lifecycle

### Initialization Phase

```python
# 1. Initialize Pygame (headless for app.py)
pygame.init()
os.environ["SDL_VIDEODRIVER"] = "dummy"  # For app.py only
screen = pygame.Surface((WIDTH, HEIGHT))

# 2. Create environment
env = FlaskRLCarEnv()

# 3. Reset to initial state
obs = env.reset()  # Returns 6D observation vector

# 4. Initialize neural networks
#    - Haiku parameters are created with dummy inputs
#    - PRNG keys are initialized for JAX randomness
```

### Training Episode

```python
episode_reward = 0
done = False

while not done:
    # 1. OBSERVE
    obs = env.get_observation()  # [5 sensors + speed]
    
    # 2. DECIDE ACTION
    if manual_control:
        action = game_state['action']  # From keyboard/UI
    else:
        latent = world_model.encode(obs)
        explore = (random.uniform() < explore_prob)
        action = actor.get_action(latent, explore=explore)
    
    # 3. EXECUTE
    next_obs, reward, done = env.step(action)
    episode_reward += reward
    
    # 4. TRAIN (every 10 steps)
    if step_counter % 10 == 0:
        world_model.add_experience(obs, action, next_obs, reward)
        loss = world_model.train_step()
        training_losses.append(loss)
    
    # 5. UPDATE STATE
    obs = next_obs
    step_counter += 1
    explore_prob *= 0.999  # Decay exploration
    
    # 6. RENDER
    frame = env.render()
    game_state['frame'] = frame  # For video streaming

# Episode complete
episode_rewards.append(episode_reward)
obs = env.reset()  # Start new episode
```

### Model Checkpointing

Automatic saves occur in `app.py` when track progress improves:

```python
current_progress = car.calculate_track_progress(track_points)
if current_progress > best_progress:
    best_progress = current_progress
    save_model()  # Saves to models/model_ep{N}.pkl
```

## Threading Model

### Thread Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Thread                           │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Flask Application                         │ │
│  │                                                        │ │
│  │  • Handles HTTP requests                              │ │
│  │  • Serves static files                                │ │
│  │  • Reads from game_state dict                         │ │
│  │  • Writes control inputs to game_state                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Game Loop Thread                         │
│                     (Daemon Thread)                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Game Loop Function                        │ │
│  │                                                        │ │
│  │  • Runs at ~30 FPS                                     │ │
│  │  • Executes RL training steps                          │ │
│  │  • Renders frames                                      │ │
│  │  • Writes to game_state dict                           │ │
│  │  • Reads control inputs from game_state                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Shared State: `game_state` Dictionary

```python
game_state = {
    'running': False,              # Control flag
    'manual_control': True,        # AI vs manual mode
    'action': [0.0, 0.0],         # Current control input
    'obs': None,                  # Latest observation
    'reward': 0,                  # Latest reward
    'done': False,                # Episode termination flag
    'frame': None,                # Latest rendered frame (base64)
    'training_stats': {
        'episode_rewards': [],
        'training_losses': [],
        'exploration_rate': 1.0
    }
}
```

**Thread Safety Notes**:
- Python GIL provides basic protection for dict operations
- `game_state['frame']` is reassigned atomically (not mutated)
- Lists in `training_stats` are only appended to (safe in CPython)
- For production, consider using `threading.Lock()` for critical sections

### Video Streaming Generator

```python
def generate():
    while True:
        if game_state['frame']:
            jpeg_bytes = base64.b64decode(game_state['frame'])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' 
                   + jpeg_bytes + b'\r\n\r\n')
        time.sleep(1/FPS)
```

This generator function runs in Flask's request thread, continuously yielding MJPEG frames.

## Data Flow

### Observation Pipeline

```
┌──────────────┐
│ Pygame World │
│  (Car, Track)│
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│   update_sensors()   │  Ray casting to measure distances
│                      │
│ • Cast 5 rays        │
│ • Check intersections│
│ • Store distances    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  get_observation()   │  Normalize to [0, 1] range
│                      │
│ sensors / 150.0      │
│ speed / 5.0          │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  6D Observation      │  [s1, s2, s3, s4, s5, speed]
│  (NumPy/JAX array)   │
└──────────────────────┘
```

### Action Selection Pipeline

```
┌──────────────────────┐
│  6D Observation      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ WorldModel.encode()  │  Compress to latent space
│                      │
│ [6] → [64] → [32]    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ 32D Latent State     │  Abstract representation
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Actor.get_action()  │  Policy network
│                      │
│ [32] → [64] → [32]   │
│      → [2 tanh]      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Add Exploration      │  If exploring
│ Noise (Gaussian)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ 2D Action            │  [steering, throttle]
│ Range: [-1.0, 1.0]   │
└──────────────────────┘
```

### Training Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     Experience Generation                     │
│                                                              │
│  (obs, action) → Environment → (next_obs, reward)           │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     Replay Buffer (10K)                      │
│                                                              │
│  deque[(obs, action, next_obs, reward), ...]                │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼ Sample batch of 32
┌──────────────────────────────────────────────────────────────┐
│                     World Model Training                     │
│                                                              │
│  1. Encode observations → latents                            │
│  2. Predict next latents from (latent, action)              │
│  3. Predict rewards from latents                             │
│  4. Decode latents → reconstructed observations              │
│  5. Compute losses (MSE on reconstruction, reward)           │
│  6. (Note: No gradient updates in current impl)              │
└──────────────────────────────────────────────────────────────┘
```

## Neural Network Architecture

### JAX + Haiku Framework

**Why JAX?**
- Hardware acceleration (GPU/TPU support)
- Automatic differentiation via `jax.grad()`
- JIT compilation for speed (`@jit` decorator)
- Functional programming paradigm

**Why Haiku?**
- High-level neural network library on JAX
- Clean separation of network definition and parameters
- `hk.transform()` converts Python functions to pure functions
- Compatible with JAX's functional approach

### Network Definition Pattern

```python
# 1. Define network as a pure function
def encoder_fn(obs):
    net = hk.Sequential([
        hk.Linear(64),
        jax.nn.relu,
        hk.Linear(32)
    ])
    return net(obs)

# 2. Transform to pure function with explicit parameters
encoder = hk.transform(encoder_fn)

# 3. Initialize parameters with dummy input
key = random.PRNGKey(42)
dummy_obs = jnp.zeros((1, obs_size))
encoder_params = encoder.init(key, dummy_obs)

# 4. Apply network (forward pass)
latent = encoder.apply(encoder_params, None, obs_batch)
```

### Parameter Management

All network parameters are stored as nested dicts/arrays:

```python
encoder_params = {
    'sequential/linear_0': {
        'w': array(...),  # Shape: (obs_size, 64)
        'b': array(...)   # Shape: (64,)
    },
    'sequential/linear_1': {
        'w': array(...),  # Shape: (64, latent_size)
        'b': array(...)   # Shape: (latent_size,)
    }
}
```

These can be serialized with `pickle` for model checkpointing.

## Model Persistence

### Save Format

```python
model_data = {
    'encoder_params': encoder_params,      # Haiku params dict
    'dynamics_params': dynamics_params,    # Haiku params dict
    'decoder_params': decoder_params,      # Haiku params dict
    'reward_params': reward_params,        # Haiku params dict
    'actor_params': actor_params,          # Haiku params dict
    'explore_prob': 0.45,                 # Current exploration rate
    'episode_rewards': [12.3, 45.6, ...], # Training history
    'training_losses': [2.1, 1.8, ...]    # Loss history
}
```

### Save Locations

**Automatic Saves** (`app.py`):
- Trigger: When `car.calculate_track_progress()` exceeds previous best
- Path: `models/model_ep{N}.pkl` where N = number of episodes completed
- Frequency: Only on improvement (not every episode)

**Manual Saves**:
- Trigger: Click "Save Model" button in UI
- Endpoint: `GET /save_model`
- Same format as automatic saves

### Load Process

```python
# 1. User selects model from UI list
# 2. GET /load_model?file=model_ep50.pkl
# 3. Load pickle file
with open(filename, 'rb') as f:
    model_data = pickle.load(f)

# 4. Restore all parameters
world_model.encoder_params = model_data['encoder_params']
world_model.dynamics_params = model_data['dynamics_params']
# ... (all network params)

# 5. Restore training state
explore_prob = model_data['explore_prob']
episode_rewards = model_data['episode_rewards']

# 6. Update UI statistics
game_state['training_stats']['episode_rewards'] = episode_rewards[-100:]
```

## Performance Considerations

### Bottlenecks

1. **Rendering**: Pygame → PIL → JPEG encoding → base64
   - Mitigation: Reduced JPEG quality to 70%, runs at 30 FPS (not 60)

2. **JAX Compilation**: First execution of JAX functions triggers XLA compilation
   - Mitigation: `JAX_DISABLE_JIT=1` in app.py to avoid compilation hangs

3. **Replay Buffer Sampling**: Random indexing into Python deque
   - Mitigation: Buffer limited to 10K experiences

### Optimization Opportunities

1. **Enable JIT**: Remove `JAX_DISABLE_JIT=1` after debugging
2. **GPU Acceleration**: JAX automatically uses CUDA if available
3. **Batch Training**: Actor training currently commented out - could be enabled
4. **Asynchronous Rendering**: Separate render thread from game loop

## Summary

The architecture follows a clean separation of concerns:

- **Simulation Layer** (Pygame): Physics and visualization
- **RL Layer** (JAX/Haiku): Learning algorithms
- **API Layer** (Flask): Remote control and monitoring
- **Persistence Layer** (Pickle): Model checkpointing

This modular design allows independent development and testing of each component while maintaining clear interfaces between them.

For deployment and operational details, see [OPERATIONS.md](OPERATIONS.md).

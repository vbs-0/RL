# API and Configuration Guide

This document describes all available REST endpoints, configuration parameters, setup procedures, and integration options for the RL Car Racing system.

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Flask REST API](#flask-rest-api)
3. [Frontend Controls](#frontend-controls)
4. [Configuration Parameters](#configuration-parameters)
5. [Model Management](#model-management)
6. [Environment Variables](#environment-variables)
7. [CLI Usage](#cli-usage)

## Installation and Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Hardware**: CPU sufficient; GPU optional for JAX acceleration
- **Display**: Not required (headless mode supported)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd rl-car-racing
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `Flask` - Web framework
- `pygame` - Simulation and rendering
- `numpy` - Numerical operations
- `jax` - Accelerated computing
- `jaxlib` - JAX backend
- `dm-haiku` - Neural network library
- `noise` - Perlin noise generation
- `Pillow` - Image processing
- `optax` - Gradient optimization

### Step 4: Verify Installation

```bash
python test_training.py
```

This runs a quick sanity check. You should see:
```
Starting training sanity check...
Episode 1 completed
Training check completed successfully!
```

### Step 5: Launch Web Server

```bash
python app.py
```

Expected output:
```
 * Serving Flask app 'app'
 * Running on http://0.0.0.0:5000
```

### Step 6: Access Dashboard

Open browser to: `http://localhost:5000`

You should see:
- Video stream placeholder
- Game control buttons
- Training statistics panel
- Model management interface

## Flask REST API

All endpoints respond with JSON (except `/video_feed`).

### GET /

**Purpose**: Serve the main HTML dashboard

**Response**: HTML page

**Example**:
```bash
curl http://localhost:5000/
```

---

### GET /video_feed

**Purpose**: Stream MJPEG video feed of game

**Response**: `multipart/x-mixed-replace` MJPEG stream

**Frame Rate**: 30 FPS

**Resolution**: 800x600 pixels

**Usage in HTML**:
```html
<img src="/video_feed" width="800" height="600">
```

**Implementation**:
- Pygame renders to surface
- Surface converted to PIL Image
- Image encoded as JPEG (quality 70%)
- JPEG encoded as base64
- Streamed with MJPEG boundaries

---

### GET /start

**Purpose**: Start the game loop

**Response**:
```json
{
  "status": "started"
}
```

**Side Effects**:
- Sets `game_state['running'] = True`
- Game loop thread begins executing steps
- Training commences if in AI mode

**Example**:
```bash
curl http://localhost:5000/start
```

---

### GET /stop

**Purpose**: Pause the game loop

**Response**:
```json
{
  "status": "stopped"
}
```

**Side Effects**:
- Sets `game_state['running'] = False`
- Game loop thread enters idle state
- Training pauses
- Current episode state preserved

**Example**:
```bash
curl http://localhost:5000/stop
```

---

### GET /reset

**Purpose**: Reset environment to initial state

**Response**:
```json
{
  "status": "reset"
}
```

**Side Effects**:
- Car repositioned to track start
- Velocity and angle reset to 0
- Sensors updated
- Current episode reward discarded

**Example**:
```bash
curl http://localhost:5000/reset
```

---

### GET /toggle_control

**Purpose**: Switch between manual and AI control

**Response**:
```json
{
  "status": "toggled",
  "manual_control": false
}
```

**Side Effects**:
- Toggles `game_state['manual_control']`
- If `manual_control=True`: Uses keyboard/API input
- If `manual_control=False`: Uses RL agent actions

**Example**:
```bash
curl http://localhost:5000/toggle_control
```

---

### POST /control

**Purpose**: Send manual control inputs

**Request Body**:
```json
{
  "steering": -0.5,
  "throttle": 1.0
}
```

**Parameters**:
- `steering` (float): -1.0 (left) to 1.0 (right)
- `throttle` (float): -1.0 (brake) to 1.0 (accelerate)

**Response**:
```json
{
  "status": "ok"
}
```

**Side Effects**:
- Updates `game_state['action']`
- Only effective when `manual_control=True`

**Example**:
```bash
curl -X POST http://localhost:5000/control \
  -H "Content-Type: application/json" \
  -d '{"steering": 0.0, "throttle": 1.0}'
```

---

### GET /stats

**Purpose**: Retrieve current training statistics

**Response**:
```json
{
  "episode_rewards": [12.5, 23.1, 45.7, ...],
  "training_losses": [2.34, 1.98, 1.65, ...],
  "exploration_rate": 0.456
}
```

**Fields**:
- `episode_rewards` (array): Last 100 episode total rewards
- `training_losses` (array): Last 100 training step losses
- `exploration_rate` (float): Current probability of exploration

**Update Frequency**: Dashboard polls every 5 seconds

**Example**:
```bash
curl http://localhost:5000/stats
```

---

### GET /save_model

**Purpose**: Manually save current model checkpoint

**Response**:
```json
{
  "status": "saved"
}
```

**Side Effects**:
- Creates file: `models/model_ep{N}.pkl`
- N = number of completed episodes
- Serializes all network parameters and training state

**Example**:
```bash
curl http://localhost:5000/save_model
```

---

### GET /load_model

**Purpose**: Load a saved model checkpoint

**Query Parameters**:
- `file` (string, required): Filename in `models/` directory

**Response** (success):
```json
{
  "status": "loaded"
}
```

**Response** (failure):
```json
{
  "status": "failed"
}
```

**Side Effects**:
- Restores all neural network parameters
- Restores exploration rate
- Restores training history
- Updates dashboard statistics

**Example**:
```bash
curl "http://localhost:5000/load_model?file=model_ep50.pkl"
```

---

### GET /list_models

**Purpose**: List all available model checkpoints

**Response**:
```json
{
  "models": [
    "model_ep0.pkl",
    "model_ep12.pkl",
    "model_ep45.pkl",
    "model_ep81.pkl"
  ]
}
```

**Sorting**: Lexicographic (not numerical)

**Example**:
```bash
curl http://localhost:5000/list_models
```

---

## Frontend Controls

### Keyboard Shortcuts

The web dashboard captures keyboard events for manual control:

| Key | Action |
|-----|--------|
| `↑` (Up Arrow) | Accelerate (throttle = 1.0) |
| `↓` (Down Arrow) | Brake (throttle = -1.0) |
| `←` (Left Arrow) | Steer left (steering = -1.0) |
| `→` (Right Arrow) | Steer right (steering = 1.0) |

**Notes**:
- Multiple keys can be pressed simultaneously
- Only effective when manual control is enabled
- Keys are sent to `/control` endpoint via POST

### Button Controls

**Start Game**
- Overlay button (initially visible)
- Calls `/start` endpoint
- Hides overlay on success

**Reset**
- Red button in control panel
- Calls `/reset` endpoint
- Restarts current episode

**Toggle AI/Manual**
- Blue button in control panel
- Calls `/toggle_control` endpoint
- Updates button text to show current mode

**Stop**
- Red button in control panel
- Calls `/stop` endpoint
- Shows overlay again

**Save Model**
- Green button in model management section
- Calls `/save_model` endpoint
- Refreshes model list on success

**Load Model**
- Blue button in model management section
- Opens model list (populated by `/list_models`)
- Click on model name to load via `/load_model`

### Visual Indicators

**On-Screen Display** (in game view):
- Speed (top-left)
- Position coordinates (top-left)
- Exploration rate (top-left)
- Track progress percentage (top-left)
- Latent state visualization (top-right, colorful squares)
- Control mode ("Manual Control" or "AI Control", bottom-right)

**Statistics Panel**:
- Exploration Rate (numeric)
- Episodes Completed (count)
- Average Reward (mean of last 100 episodes)
- Average Loss (mean of last 100 training steps)
- Episode Rewards Chart (bar chart)
- Training Loss Chart (bar chart)

## Configuration Parameters

Configuration is currently hardcoded. To customize, edit the following variables:

### Global Settings (`app.py`)

```python
WIDTH = 800          # Screen width in pixels
HEIGHT = 600         # Screen height in pixels
FPS = 30            # Frames per second for game loop
```

### Car Physics (`test.py` - Car class)

```python
max_speed = 5.0          # Maximum velocity
size = 20                # Car radius for rendering
sensor_length = 150      # Maximum sensor range in pixels
sensor_angles = [-45, -22.5, 0, 22.5, 45]  # Sensor directions in degrees
```

### Track Generation (`test.py` - Track class)

```python
seed = 42                   # Random seed for Perlin noise
num_points = 100            # Track segments
radius = min(WIDTH, HEIGHT) * 0.35  # Base track radius
track_width = 40            # Track width in pixels
checkpoint_interval = 0.1   # Progress between checkpoints (10%)
num_obstacles = 5           # Number of yellow obstacles
```

### World Model (`test.py` - WorldModel class)

```python
obs_size = 6                # Observation dimensions (5 sensors + speed)
action_size = 2             # Action dimensions (steering + throttle)
latent_size = 32            # Latent space dimensions
buffer_maxlen = 10000       # Replay buffer size
batch_size = 32             # Training batch size
```

### Training Hyperparameters (`app.py` - FlaskRLCarEnv)

```python
train_every = 10            # Train every N steps
explore_prob = 1.0          # Initial exploration rate
min_explore_prob = 0.1      # Minimum exploration rate
explore_decay = 0.999       # Exploration decay per step
```

### Reward Shaping (`app.py` - calculate_reward method)

```python
crash_penalty = -10.0       # Penalty for going off track
speed_reward = speed        # Reward proportional to speed
progress_reward = 5.0       # Reward for passing checkpoint
time_bonus = 1000 / dt      # Time-based bonus for fast checkpoints
survival_reward = 0.1       # Small reward per step on track
```

### Customization Example

To increase exploration duration, edit `app.py`:

```python
# Before
self.explore_decay = 0.999      # Reaches 0.1 after ~2300 steps

# After
self.explore_decay = 0.9995     # Reaches 0.1 after ~4600 steps
```

To change neural network sizes, edit `test.py`:

```python
# In WorldModel.__init__
def encoder_fn(obs):
    net = hk.Sequential([
        hk.Linear(128),  # Increased from 64
        jax.nn.relu,
        hk.Linear(64)    # Increased from 32 (latent_size must match)
    ])
    return net(obs)

# Update latent_size
self.latent_size = 64  # Was 32
```

**Note**: Changing network architecture requires retraining from scratch. Old checkpoints will be incompatible.

## Model Management

### Checkpoint Format

Models are saved as Python pickle files (`.pkl`) containing:

```python
{
    'encoder_params': {...},      # Haiku parameter tree
    'dynamics_params': {...},
    'decoder_params': {...},
    'reward_params': {...},
    'actor_params': {...},
    'explore_prob': 0.234,        # Training state
    'episode_rewards': [...],     # History
    'training_losses': [...]
}
```

### Automatic Checkpointing

**Trigger**: Track progress improvement

When the car reaches a new personal best for track progress:
```python
if current_progress > best_progress:
    best_progress = current_progress
    save_model()
```

**Naming Convention**: `model_ep{N}.pkl`
- N = number of episodes completed at save time
- Example: `model_ep45.pkl` saved after 45th episode

**Directory**: `models/` (created automatically)

### Manual Checkpointing

**Web UI**: Click "Save Model" button

**API**: `GET /save_model`

**Python**:
```python
env = FlaskRLCarEnv()
env.save_model()
```

### Loading Checkpoints

**Web UI**:
1. Click "Load Model" button
2. Select checkpoint from list
3. Confirm load

**API**:
```bash
curl "http://localhost:5000/load_model?file=model_ep50.pkl"
```

**Python**:
```python
env = FlaskRLCarEnv()
success = env.load_model('models/model_ep50.pkl')
```

### Checkpoint Compatibility

Checkpoints are compatible if:
- Network architectures match (layer sizes)
- Input/output dimensions match
- Same version of JAX/Haiku

Incompatibility causes:
- Shape mismatch errors
- Key errors in parameter dicts

**Best Practice**: Save training configuration with each checkpoint

### Versioning Strategy

Since filenames include episode count, you can:

1. **Track Progress Over Time**:
   ```
   model_ep0.pkl   → Initial random policy
   model_ep10.pkl  → Early learning
   model_ep50.pkl  → Intermediate skill
   model_ep200.pkl → Advanced driving
   ```

2. **A/B Testing**:
   - Load `model_ep100.pkl`
   - Observe performance
   - Load `model_ep150.pkl`
   - Compare metrics

3. **Rollback on Regression**:
   - If training degrades, load earlier checkpoint
   - Continue training with adjusted hyperparameters

## Environment Variables

### JAX Configuration

**JAX_DISABLE_JIT** (default: `'1'` in app.py)
```bash
export JAX_DISABLE_JIT=1  # Disable JIT compilation
```
- Use: Prevents initialization hangs during development
- Trade-off: Slower execution, easier debugging
- Production: Set to `0` for performance

**XLA_PYTHON_CLIENT_PREALLOCATE** (default: `'false'`)
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```
- Use: Prevents JAX from allocating all GPU memory upfront
- Benefit: Allows other processes to use GPU

### Pygame Configuration

**SDL_VIDEODRIVER** (default: `'dummy'` in app.py)
```bash
export SDL_VIDEODRIVER=dummy  # Headless mode
```
- Use: Runs Pygame without display (required for servers)
- For local testing with display, unset or use `'x11'`

### Flask Configuration

**FLASK_ENV** (optional)
```bash
export FLASK_ENV=development  # Enables debug mode
```
- Auto-reloads on code changes
- Detailed error pages
- Not recommended for production

**FLASK_DEBUG** (optional)
```bash
export FLASK_DEBUG=1  # Enables debugger
```

## CLI Usage

### Running Different Components

**Web Server (Production)**:
```bash
python app.py
# Starts Flask + headless game loop
# Access: http://localhost:5000
```

**Standalone Game (Development)**:
```bash
python game.py
# Opens Pygame window
# Manual control: Arrow keys
# Toggle AI: Spacebar
# Reset: R key
```

**Enhanced Training (Development)**:
```bash
python test.py
# Same as game.py but with Optax optimizers
```

**Simple Demo**:
```bash
python car_game.py
# Basic manual-only version for learning
```

**Training Sanity Check**:
```bash
python test_training.py
# Quick validation of JAX/Haiku setup
```

### Custom Ports

To run Flask on a different port:

```bash
# Edit app.py, change:
app.run(host='0.0.0.0', port=8080, threaded=True)
```

Or set via environment:
```bash
export FLASK_RUN_PORT=8080
python app.py
```

### Background Execution

Run server in background:
```bash
nohup python app.py > server.log 2>&1 &
```

Check logs:
```bash
tail -f server.log
```

Stop server:
```bash
pkill -f "python app.py"
```

### Docker Deployment (Example)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV SDL_VIDEODRIVER=dummy
ENV JAX_DISABLE_JIT=0

EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t rl-car-racing .
docker run -p 5000:5000 rl-car-racing
```

## Integration Examples

### Python Client

```python
import requests
import json

BASE_URL = 'http://localhost:5000'

# Start game
requests.get(f'{BASE_URL}/start')

# Enable AI control
response = requests.get(f'{BASE_URL}/toggle_control')
print(response.json())  # {'status': 'toggled', 'manual_control': False}

# Monitor training
while True:
    stats = requests.get(f'{BASE_URL}/stats').json()
    print(f"Episodes: {len(stats['episode_rewards'])}, "
          f"Avg Reward: {sum(stats['episode_rewards'])/len(stats['episode_rewards']):.2f}")
    time.sleep(10)
```

### JavaScript Client

```javascript
// Send manual control input
async function driveForward() {
    await fetch('/control', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({steering: 0, throttle: 1.0})
    });
}

// Poll stats
setInterval(async () => {
    const response = await fetch('/stats');
    const stats = await response.json();
    console.log('Exploration Rate:', stats.exploration_rate);
}, 5000);
```

### cURL Scripts

```bash
#!/bin/bash
# auto_train.sh - Automated training session

BASE_URL="http://localhost:5000"

echo "Starting training session..."
curl -s "$BASE_URL/start" > /dev/null

echo "Enabling AI control..."
curl -s "$BASE_URL/toggle_control" > /dev/null

echo "Training for 1 hour..."
sleep 3600

echo "Saving model..."
curl -s "$BASE_URL/save_model"

echo "Stopping..."
curl -s "$BASE_URL/stop"

echo "Done!"
```

## Troubleshooting

### Port Already in Use

```
OSError: [Errno 98] Address already in use
```

**Solution**:
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

### JAX/CUDA Issues

```
RuntimeError: CUDA initialization error
```

**Solution**:
```bash
# Check CUDA availability
python -c "import jax; print(jax.devices())"

# Force CPU-only
export JAX_PLATFORM_NAME=cpu
```

### Pygame Display Errors

```
pygame.error: No available video device
```

**Solution**:
```bash
# Ensure dummy driver is set
export SDL_VIDEODRIVER=dummy
python app.py
```

### Model Load Failures

```
KeyError: 'encoder_params'
```

**Solution**:
- Checkpoint file corrupted or from incompatible version
- Try different checkpoint
- Retrain from scratch if necessary

---

For operational best practices and deployment strategies, see [OPERATIONS.md](OPERATIONS.md).

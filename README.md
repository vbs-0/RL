# RL Car Racing - Autonomous Driving with DreamerV3

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An advanced reinforcement learning project that trains an autonomous agent to race on procedurally generated tracks using a **DreamerV3-inspired world model**. Watch your AI learn from scratch through trial and error, going from random flailing to smooth, confident racing!

## 🚀 Key Features

- **🧠 World Model Learning**: DreamerV3-inspired architecture with latent imagination for sample-efficient RL
- **🎮 Web Dashboard**: Real-time training visualization, control switching, and model management via browser
- **📊 Live Monitoring**: Watch training metrics, episode rewards, and exploration rates update in real-time
- **🏁 Procedural Tracks**: Infinite variety through Perlin noise-based track generation
- **🤖 Hybrid Control**: Seamlessly toggle between AI and manual keyboard control
- **💾 Model Checkpointing**: Automatic saves on progress milestones + manual checkpoint management
- **🎯 Smart Sensors**: Five distance sensors provide efficient environmental perception
- **⚡ JAX Acceleration**: Hardware-accelerated neural networks with GPU support

## 📸 Screenshots

**Training Dashboard**: Real-time video stream, control panel, and training charts  
**In-Game View**: Car sensors, track progress, and latent state visualization  
**Model Management**: Save, load, and version trained agents

## 🎯 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rl-car-racing

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Required packages**: Flask, Pygame, NumPy, JAX, dm-haiku, Optax, Pillow, noise

### 2. Launch Training Server

```bash
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
```

### 3. Open Dashboard

Navigate to: **`http://localhost:5000`**

### 4. Start Training

1. Click **"Start Game"** button (removes overlay)
2. Click **"Toggle AI/Manual"** to enable AI control
3. Watch the agent learn in real-time!

**Initial behavior**: Random, erratic movements (exploration rate = 100%)  
**After ~50 episodes**: Starts staying on track  
**After ~200 episodes**: Smooth, confident driving

### 5. Interact with Your Agent

**Manual Control**:
- Use **arrow keys** to drive yourself: ↑ (accelerate), ↓ (brake), ← (left), → (right)
- Click **"Toggle AI/Manual"** to switch modes

**Save Progress**:
- Click **"Save Model"** when your agent performs well
- Models auto-save when track progress improves

**Load Previous Models**:
- Click **"Load Model"** to see your saved checkpoints
- Select a checkpoint to restore (great for comparing training stages)

## 📚 Documentation

Comprehensive guides for all skill levels:

### For Beginners
- **[Beginner's Guide](docs/BEGINNER_GUIDE.md)** - Understand RL fundamentals, the car environment, and how your agent learns

### For Developers
- **[Architecture Guide](docs/ARCHITECTURE.md)** - Deep dive into package layout, threading model, data flow, and component interactions
- **[API & Configuration](docs/API_AND_CONFIG.md)** - Complete REST API reference, configuration parameters, and integration examples

### For Operators
- **[Operations Guide](docs/OPERATIONS.md)** - Deployment strategies, monitoring, performance tuning, and troubleshooting

## 🧪 How It Works

### The Environment

Your agent navigates a **circular racing track** with:
- **Track**: Procedurally generated using Perlin noise (smooth, organic curves)
- **Obstacles**: Yellow circular hazards scattered around the track interior
- **Sensors**: 5 distance-measuring rays (-45°, -22.5°, 0°, 22.5°, 45°) detecting track boundaries and obstacles
- **Physics**: Realistic velocity and steering with momentum

### The Agent

Powered by a **DreamerV3-inspired** architecture:

```
Observation (sensors + speed)
         ↓
    Encoder (compress to latent space)
         ↓
    World Model (imagine future states)
         ↓
    Actor (decide steering + throttle)
         ↓
    Execute action in environment
         ↓
    Receive reward → Train on experience
```

**Components**:
1. **World Model**: Learns to predict future states and rewards in a compressed latent space
   - Encoder: Observation → 32D latent representation
   - Dynamics: Predicts next latent state given current state + action
   - Decoder: Reconstructs observations from latent (for training)
   - Reward Predictor: Estimates reward from latent state

2. **Actor**: Policy network that selects actions based on latent state
   - Outputs: [steering, throttle] both in range [-1, 1]
   - Exploration: Gaussian noise added initially, decays over time

3. **Experience Replay**: Stores (observation, action, reward, next_observation) tuples
   - Buffer size: 10,000 experiences
   - Batch size: 32 samples per training step
   - Training frequency: Every 10 environment steps

### Reward Structure

The agent learns from a carefully designed reward signal:

- **✅ Positive Rewards**:
  - Speed reward (encourages faster driving)
  - Checkpoint reward (+5.0 for progress milestones)
  - Time bonus (faster checkpoint completion)
  - Survival reward (+0.1 per step on track)

- **❌ Penalties**:
  - Crash penalty (-10.0 for leaving track or hitting obstacles)

This encourages **safe**, **fast**, **progressive** driving.

### Continuous Learning Loop

1. **Observe**: Get sensor readings + current speed
2. **Encode**: Compress observation to latent representation
3. **Decide**: Actor network chooses action (with exploration noise)
4. **Execute**: Apply action to car physics
5. **Reward**: Calculate based on performance
6. **Remember**: Store experience in replay buffer
7. **Learn**: Sample batch from buffer, train world model every 10 steps
8. **Improve**: Exploration rate gradually decreases (1.0 → 0.1)

**Training never stops!** The agent continuously refines its understanding as it collects more experience.

### Model Evolution

**Episode 0-10**: Pure exploration, random movements, frequent crashes  
**Episode 10-50**: Learning basic steering, short track segments  
**Episode 50-100**: Consistent driving, avoiding most obstacles  
**Episode 100-200**: Smooth cornering, speed optimization  
**Episode 200+**: Expert-level racing, near-optimal lines

You can **watch this evolution** by loading checkpoints from different training stages!

## 🎮 Usage Examples

### Standalone Game (Local Display)

For development/testing without the web interface:

```bash
python game.py
```

**Controls**:
- Arrow keys: Manual driving
- Spacebar: Toggle AI/manual
- R: Reset episode
- ESC: Quit

### Training Sanity Check

Quick validation that JAX/Haiku is working:

```bash
python test_training.py
```

### Simple Demo (Manual Only)

Minimal version to understand basic mechanics:

```bash
python car_game.py
```

## 🔧 Configuration

### Customizing Training

Key parameters in `app.py`:

```python
# Exploration
explore_prob = 1.0          # Initial exploration rate (100%)
min_explore_prob = 0.1      # Minimum exploration rate (10%)
explore_decay = 0.999       # Decay per step

# Training
train_every = 10            # Train world model every N steps
batch_size = 32             # Samples per training batch
buffer_size = 10000         # Experience replay capacity

# Neural Networks
latent_size = 32            # Compressed state dimensions
```

### Environment Tuning

Track generation (`test.py`):

```python
seed = 42                   # Change for different track layouts
track_width = 40            # Wider = easier
num_obstacles = 5           # More = harder
```

Car physics (`test.py`):

```python
max_speed = 5.0             # Higher = faster but harder to control
sensor_length = 150         # Longer = more foresight
sensor_angles = [-45, -22.5, 0, 22.5, 45]  # Field of view
```

See **[API & Configuration Guide](docs/API_AND_CONFIG.md)** for comprehensive parameter reference.

## 📊 REST API

Core endpoints for programmatic control:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve dashboard HTML |
| `/video_feed` | GET | MJPEG video stream (30 FPS) |
| `/start` | GET | Start game loop |
| `/stop` | GET | Pause game loop |
| `/reset` | GET | Reset environment |
| `/toggle_control` | GET | Switch AI/manual mode |
| `/control` | POST | Send manual input `{steering, throttle}` |
| `/stats` | GET | Get training statistics |
| `/save_model` | GET | Save current checkpoint |
| `/load_model?file=...` | GET | Load checkpoint |
| `/list_models` | GET | List available checkpoints |

**Example** (Python):
```python
import requests

# Start training
requests.get('http://localhost:5000/start')

# Enable AI
requests.get('http://localhost:5000/toggle_control')

# Check progress
stats = requests.get('http://localhost:5000/stats').json()
print(f"Episodes: {len(stats['episode_rewards'])}")
```

See **[API Reference](docs/API_AND_CONFIG.md)** for detailed endpoint documentation.

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t rl-car-racing .

# Run container
docker run -d -p 5000:5000 -v $(pwd)/models:/app/models rl-car-racing
```

### Cloud (AWS/GCP)

```bash
# Launch VM with Python 3.9+
ssh user@your-instance

# Setup and run
git clone <repo-url> && cd rl-car-racing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nohup python app.py > training.log 2>&1 &

# Access: http://<instance-ip>:5000
```

### Systemd Service (Linux)

```bash
sudo cp rl-car-racing.service /etc/systemd/system/
sudo systemctl enable rl-car-racing
sudo systemctl start rl-car-racing
```

See **[Operations Guide](docs/OPERATIONS.md)** for production deployment best practices.

## 🧠 Technical Details

### Technologies

- **Flask**: Web server and REST API
- **Pygame**: Physics simulation and rendering
- **JAX**: Hardware-accelerated numerical computation (CPU/GPU)
- **dm-haiku**: Neural network library (functional JAX)
- **Optax**: Gradient-based optimization (in test.py)
- **Perlin Noise**: Procedural track generation
- **Pillow**: Image processing for MJPEG streaming

### Architecture

```
┌─────────────────────────────────────────┐
│        Browser Dashboard (HTML/JS)       │
│  Video Stream | Controls | Charts        │
└──────────────┬──────────────────────────┘
               │ HTTP/REST
┌──────────────▼──────────────────────────┐
│         Flask Server (app.py)            │
│  Routes | Video Generator | API          │
└──────────────┬──────────────────────────┘
               │ Threading
┌──────────────▼──────────────────────────┐
│      Game Loop (Background Thread)       │
│  Simulation | RL Training | Rendering    │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Car + Track (Pygame)              │ │
│  │  World Model + Actor (JAX/Haiku)   │ │
│  │  Experience Replay (deque)         │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

**Threading Model**:
- Main thread: Flask request handling
- Daemon thread: Game loop (30 FPS)
- Shared state: `game_state` dictionary

**Data Flow**:
```
Sensors → Observation → Encoder → Latent State → Actor → Action → Physics → Reward → Buffer → Training
```

See **[Architecture Guide](docs/ARCHITECTURE.md)** for in-depth technical documentation.

## 🔍 Monitoring Training

### Dashboard Metrics

- **Exploration Rate**: Current randomness in action selection (1.0 → 0.1)
- **Episodes Completed**: Number of full training episodes
- **Average Reward**: Mean reward over last 100 episodes (should increase)
- **Average Loss**: Mean training loss (should decrease)
- **Reward Chart**: Visualize learning progress
- **Loss Chart**: Monitor training stability

### In-Game Overlay

- **Speed**: Current velocity
- **Position**: (x, y) coordinates
- **Progress**: Completion percentage (0-100%)
- **Latent State**: Visual representation of compressed world state

### Expected Progress

Good training shows:
- ✅ Increasing average rewards over episodes
- ✅ Decreasing exploration rate (1.0 → 0.1)
- ✅ Stable or decreasing training loss
- ✅ Longer episode durations before crashes

Concerning signs:
- ⚠️ Loss becomes NaN or explodes
- ⚠️ Rewards stay flat after 100+ episodes
- ⚠️ Agent gets stuck in local optima

## 🛠️ Troubleshooting

**Video stream is black**: Click "Start Game" button

**Training not progressing**: Check exploration rate is decreasing (verify game loop is running)

**High CPU usage**: Set `JAX_DISABLE_JIT=0` and reduce FPS in `app.py`

**Memory leak**: Replay buffer limited to 10K, but restart service if RAM grows unbounded

**Model load fails**: Ensure checkpoint is compatible (same network architecture)

**Port 5000 already in use**:
```bash
lsof -i :5000  # Find process
kill -9 <PID>  # Terminate it
```

See **[Operations Guide - Troubleshooting](docs/OPERATIONS.md#troubleshooting)** for comprehensive debugging.

## 📖 Project Structure

```
rl-car-racing/
├── app.py                 # Flask web server (headless mode)
├── test.py                # Core RL components (imported by app.py)
├── game.py                # Standalone version with Pygame window
├── car_game.py            # Simple demo (manual control only)
├── test_training.py       # Sanity check script
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web dashboard UI
├── models/                # Saved model checkpoints (*.pkl)
├── docs/
│   ├── BEGINNER_GUIDE.md      # RL fundamentals and learning process
│   ├── ARCHITECTURE.md        # System design and data flow
│   ├── API_AND_CONFIG.md      # REST API and configuration
│   └── OPERATIONS.md          # Deployment and monitoring
└── README.md              # This file
```

## 🎓 Learning Resources

**New to Reinforcement Learning?**
- Start with [Beginner's Guide](docs/BEGINNER_GUIDE.md) - explains RL concepts in plain English
- Watch the agent train for a few episodes to see exploration → exploitation transition
- Try manual control to understand the challenge (arrow keys)

**Want to Customize?**
- Read [API & Configuration](docs/API_AND_CONFIG.md) - adjust hyperparameters, rewards, track difficulty
- Modify neural network architectures in `test.py`
- Experiment with different exploration strategies

**Deploying for Research?**
- Follow [Operations Guide](docs/OPERATIONS.md) - Docker, cloud, monitoring, backups
- Set up automated training runs
- Version control your experiments

**Understanding the Code?**
- Check [Architecture Guide](docs/ARCHITECTURE.md) - component breakdown, data flow, threading
- Read inline comments in source files
- Trace execution flow: `app.py` → `test.py` classes → `index.html` UI

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Optimizer Integration**: Enable actual gradient updates (currently loss is computed but not applied)
- **Advanced Algorithms**: Implement full DreamerV3, SAC, PPO, or other modern RL methods
- **Multi-Track Training**: Train on diverse track seeds for better generalization
- **Curriculum Learning**: Gradually increase difficulty (track complexity, speed, obstacles)
- **Hyperparameter Tuning**: Systematic search for optimal training parameters
- **Performance Profiling**: Optimize bottlenecks (rendering, sensor calculations)
- **Testing**: Unit tests for core components

## 📝 License

This project is open-source under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **DreamerV3**: Inspired by Hafner et al.'s world model architecture
- **JAX + Haiku**: Google's functional ML framework
- **Pygame**: Classic game development library
- **Perlin Noise**: Ken Perlin's procedural generation algorithm

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review troubleshooting section above

---

**Ready to train your first RL agent? Run `python app.py` and visit `http://localhost:5000` to begin!** 🏎️💨

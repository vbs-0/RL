# Operations Guide

This document covers deployment, monitoring, maintenance, and troubleshooting practices for running the RL Car Racing system in production or research environments.

## Table of Contents
1. [Deployment Strategies](#deployment-strategies)
2. [System Requirements](#system-requirements)
3. [Monitoring and Metrics](#monitoring-and-metrics)
4. [Model Management](#model-management)
5. [Performance Tuning](#performance-tuning)
6. [Backup and Recovery](#backup-and-recovery)
7. [Scaling Considerations](#scaling-considerations)
8. [Security Best Practices](#security-best-practices)
9. [Troubleshooting](#troubleshooting)

## Deployment Strategies

### Local Development

**Use Case**: Testing, experimentation, rapid iteration

**Setup**:
```bash
# Clone repository
git clone <repo-url> && cd rl-car-racing

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

**Access**: `http://localhost:5000`

**Advantages**:
- Quick startup
- Full debugging capabilities
- Direct file system access

**Limitations**:
- Single user
- Not accessible remotely without port forwarding
- No process management

---

### Server Deployment

**Use Case**: Remote training, team collaboration, persistent experiments

**Option A: Systemd Service (Linux)**

Create `/etc/systemd/system/rl-car-racing.service`:
```ini
[Unit]
Description=RL Car Racing Training Server
After=network.target

[Service]
Type=simple
User=rluser
WorkingDirectory=/opt/rl-car-racing
Environment="PATH=/opt/rl-car-racing/.venv/bin"
Environment="SDL_VIDEODRIVER=dummy"
Environment="JAX_DISABLE_JIT=0"
ExecStart=/opt/rl-car-racing/.venv/bin/python app.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rl-car-racing
sudo systemctl start rl-car-racing
sudo systemctl status rl-car-racing
```

**View logs**:
```bash
sudo journalctl -u rl-car-racing -f
```

**Option B: Supervisor (Cross-Platform)**

Install Supervisor:
```bash
pip install supervisor
```

Create `/etc/supervisor/conf.d/rl-car-racing.conf`:
```ini
[program:rl-car-racing]
command=/opt/rl-car-racing/.venv/bin/python app.py
directory=/opt/rl-car-racing
user=rluser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/rl-car-racing/output.log
environment=SDL_VIDEODRIVER="dummy",JAX_DISABLE_JIT="0"
```

**Start**:
```bash
supervisorctl reread
supervisorctl update
supervisorctl start rl-car-racing
```

---

### Docker Deployment

**Use Case**: Reproducible environments, cloud deployment, isolation

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Environment variables
ENV SDL_VIDEODRIVER=dummy
ENV JAX_DISABLE_JIT=0
ENV PYTHONUNBUFFERED=1

# Expose Flask port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

**Build and run**:
```bash
# Build image
docker build -t rl-car-racing:latest .

# Run container
docker run -d \
  --name rl-car-racing \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  rl-car-racing:latest

# View logs
docker logs -f rl-car-racing

# Stop
docker stop rl-car-racing
```

**Docker Compose** (`docker-compose.yml`):
```yaml
version: '3.8'

services:
  rl-car-racing:
    build: .
    container_name: rl-car-racing
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - SDL_VIDEODRIVER=dummy
      - JAX_DISABLE_JIT=0
      - JAX_PLATFORM_NAME=cpu  # or 'gpu' if CUDA available
    restart: unless-stopped
```

**Usage**:
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

### Cloud Deployment

**Use Case**: Scalable infrastructure, GPU acceleration, high availability

**AWS EC2 Example**:

1. **Launch Instance**:
   - AMI: Ubuntu 20.04 LTS (or Deep Learning AMI for GPU)
   - Instance Type: `t3.medium` (CPU) or `g4dn.xlarge` (GPU)
   - Security Group: Allow inbound TCP 5000

2. **Setup**:
   ```bash
   # SSH into instance
   ssh -i key.pem ubuntu@<instance-ip>
   
   # Install dependencies
   sudo apt update
   sudo apt install python3-pip python3-venv git -y
   
   # Clone and setup
   git clone <repo-url> rl-car-racing
   cd rl-car-racing
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   
   # Run with nohup
   nohup python app.py > training.log 2>&1 &
   ```

3. **Access**: `http://<instance-public-ip>:5000`

**Google Cloud Platform (GCP)**:

Similar process using Compute Engine:
- Use `n1-standard-2` or `n1-highmem-2` for CPU
- Use `n1-standard-4` with NVIDIA T4 GPU for acceleration

**Kubernetes Deployment**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rl-car-racing
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rl-car-racing
  template:
    metadata:
      labels:
        app: rl-car-racing
    spec:
      containers:
      - name: rl-car-racing
        image: rl-car-racing:latest
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: models
          mountPath: /app/models
        env:
        - name: SDL_VIDEODRIVER
          value: "dummy"
        - name: JAX_DISABLE_JIT
          value: "0"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: rl-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: rl-car-racing-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 5000
  selector:
    app: rl-car-racing
```

## System Requirements

### Minimum Requirements

**Hardware**:
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Storage: 1 GB (code + small model collection)
- Network: 10 Mbps (for video streaming)

**Software**:
- OS: Linux (Ubuntu 20.04+), macOS (10.14+), Windows 10+
- Python: 3.8+
- Display: Not required (headless mode)

**Performance**: ~10-15 FPS training speed, suitable for experimentation

---

### Recommended Requirements

**Hardware**:
- CPU: 4+ cores, 3.0+ GHz
- RAM: 8 GB
- Storage: 5 GB (code + extensive model collection)
- Network: 50 Mbps (smooth video streaming)

**Software**:
- OS: Linux (Ubuntu 22.04 LTS)
- Python: 3.9
- GPU: Optional, JAX will auto-detect CUDA

**Performance**: 30 FPS training speed, responsive UI

---

### GPU Acceleration (Optional)

**Benefits**:
- 5-10x faster neural network forward/backward passes
- Enables larger batch sizes
- Allows more complex network architectures

**Requirements**:
- NVIDIA GPU (GTX 1060 or better)
- CUDA 11.0+
- cuDNN 8.0+

**Setup**:
```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU detection
python -c "import jax; print(jax.devices())"
# Expected: [GpuDevice(id=0)]
```

**Enable GPU in app.py**:
```python
# Remove or set to 0 to enable JIT + GPU
os.environ['JAX_DISABLE_JIT'] = '0'
```

## Monitoring and Metrics

### Built-in Dashboard Metrics

**Real-Time Metrics** (visible in UI):
- **Exploration Rate**: Current exploration probability (0.1 to 1.0)
- **Episodes Completed**: Total number of finished episodes
- **Average Reward**: Mean reward of last 100 episodes
- **Average Loss**: Mean training loss of last 100 steps
- **Episode Rewards Chart**: Bar chart of recent episode rewards
- **Training Loss Chart**: Bar chart of recent training losses

**In-Game Overlay**:
- Speed (pixels per frame)
- Position (x, y coordinates)
- Track Progress (0.0 to 1.0)
- Latent State (visual representation of compressed state)

### Logging Best Practices

**Add Structured Logging to app.py**:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# In game loop
logger.info(f"Episode {len(episode_rewards)} completed: "
            f"reward={episode_reward:.2f}, "
            f"progress={best_progress:.2f}")

# On model save
logger.info(f"Model saved: models/model_ep{len(episode_rewards)}.pkl")
```

### External Monitoring Tools

**Prometheus + Grafana**:

Expose metrics endpoint (`app.py`):
```python
from prometheus_client import Counter, Gauge, generate_latest

episode_counter = Counter('rl_episodes_total', 'Total episodes')
reward_gauge = Gauge('rl_reward_latest', 'Latest episode reward')
loss_gauge = Gauge('rl_loss_latest', 'Latest training loss')

@app.route('/metrics')
def metrics():
    return generate_latest()

# Update in game loop
episode_counter.inc()
reward_gauge.set(episode_reward)
loss_gauge.set(training_loss)
```

**TensorBoard** (requires modification):

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/rl_car_racing')

# In training loop
writer.add_scalar('Reward/episode', episode_reward, episode_num)
writer.add_scalar('Loss/training', training_loss, step_counter)
writer.add_scalar('Exploration/rate', explore_prob, step_counter)
```

Launch TensorBoard:
```bash
tensorboard --logdir=runs
```

### Health Checks

**Endpoint**: Add to `app.py`
```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'game_running': game_state['running'],
        'episodes_completed': len(env.episode_rewards),
        'uptime_seconds': time.time() - start_time
    })
```

**Monitoring Script**:
```bash
#!/bin/bash
while true; do
    status=$(curl -s http://localhost:5000/health | jq -r '.status')
    if [ "$status" != "healthy" ]; then
        echo "ALERT: Service unhealthy!"
        # Send notification (email, Slack, etc.)
    fi
    sleep 60
done
```

## Model Management

### Versioning Strategy

**Semantic Versioning for Experiments**:

```
models/
├── v1.0/
│   ├── model_ep0.pkl
│   ├── model_ep50.pkl
│   └── model_ep100.pkl
├── v1.1/  # Different hyperparameters
│   ├── model_ep0.pkl
│   └── model_ep75.pkl
└── v2.0/  # Different architecture
    └── model_ep0.pkl
```

**Metadata Tracking**:

Create `models/metadata.json`:
```json
{
  "v1.0": {
    "description": "Baseline DreamerV3 implementation",
    "hyperparameters": {
      "explore_decay": 0.999,
      "latent_size": 32,
      "train_every": 10
    },
    "started": "2024-01-15T10:00:00Z",
    "best_model": "model_ep100.pkl",
    "best_reward": 245.3
  },
  "v1.1": {
    "description": "Increased exploration duration",
    "hyperparameters": {
      "explore_decay": 0.9995,
      "latent_size": 32,
      "train_every": 10
    },
    "started": "2024-01-16T14:00:00Z"
  }
}
```

### Automated Backup

**Cron Job** (runs daily at 2 AM):
```bash
# Add to crontab: crontab -e
0 2 * * * /opt/rl-car-racing/backup_models.sh
```

**backup_models.sh**:
```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/rl-car-racing/$TIMESTAMP"
SOURCE_DIR="/opt/rl-car-racing/models"

mkdir -p $BACKUP_DIR
cp -r $SOURCE_DIR/* $BACKUP_DIR/

# Compress
tar -czf "$BACKUP_DIR.tar.gz" -C /backups/rl-car-racing $TIMESTAMP
rm -rf $BACKUP_DIR

# Keep only last 30 days
find /backups/rl-car-racing -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### Cloud Storage Sync

**AWS S3**:
```bash
# Install AWS CLI
pip install awscli

# Configure
aws configure

# Sync models to S3 (every hour)
aws s3 sync /opt/rl-car-racing/models s3://rl-car-racing-models/

# Restore from S3
aws s3 sync s3://rl-car-racing-models/ /opt/rl-car-racing/models/
```

**Google Cloud Storage**:
```bash
# Install gsutil
pip install gsutil

# Sync to GCS
gsutil -m rsync -r /opt/rl-car-racing/models gs://rl-car-racing-models/

# Restore
gsutil -m rsync -r gs://rl-car-racing-models/ /opt/rl-car-racing/models/
```

## Performance Tuning

### Profiling

**Identify Bottlenecks**:

```python
import cProfile
import pstats

# Profile game loop
profiler = cProfile.Profile()
profiler.enable()

# Run for 1000 steps
for _ in range(1000):
    env.step(action)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Common Bottlenecks**:
1. **Rendering**: Pygame → PIL → JPEG encoding
2. **Sensor Updates**: Ray casting calculations
3. **Neural Network Forward Pass**: Without JIT/GPU

### Optimization Techniques

**1. Enable JAX JIT Compilation**:

```python
# In app.py, change:
os.environ['JAX_DISABLE_JIT'] = '0'  # Was '1'
```

**Expected Speedup**: 2-5x faster neural network operations

**2. Reduce Frame Rate for Training**:

```python
# In app.py, change:
FPS = 60  # To 30 or 15 for faster training (less rendering overhead)
```

**3. Optimize JPEG Encoding**:

```python
# In FlaskRLCarEnv.render(), change:
img.save(buffer, format='JPEG', quality=50)  # Was 70
```

Lower quality = faster encoding + smaller stream size

**4. Batch Sensor Calculations**:

Vectorize ray casting using NumPy/JAX operations instead of Python loops.

**5. Use Larger Training Batches**:

```python
# In WorldModel.__init__, change:
self.batch_size = 64  # Was 32
```

Requires more RAM but more efficient GPU utilization.

### Resource Limits

**Limit CPU Usage** (Linux):
```bash
# Using nice (lower priority)
nice -n 10 python app.py

# Using cpulimit (cap at 50% of one core)
cpulimit -l 50 -p $(pgrep -f "python app.py")
```

**Limit Memory** (Docker):
```bash
docker run -d \
  --memory="2g" \
  --memory-swap="2g" \
  -p 5000:5000 \
  rl-car-racing:latest
```

## Backup and Recovery

### Disaster Recovery Plan

**Backup Frequency**:
- **Models**: Daily (automated)
- **Code**: On every commit (Git)
- **Configuration**: Weekly (manual)
- **Logs**: Rotate weekly, archive monthly

**Recovery Procedures**:

**Scenario 1: Corrupted Model**
```bash
# Stop training
curl http://localhost:5000/stop

# Restore from backup
cp /backups/rl-car-racing/20240115_020000.tar.gz .
tar -xzf 20240115_020000.tar.gz
cp 20240115_020000/* models/

# Load previous checkpoint via UI or API
curl "http://localhost:5000/load_model?file=model_ep50.pkl"

# Resume training
curl http://localhost:5000/start
```

**Scenario 2: Code Rollback**
```bash
# Identify last working commit
git log --oneline

# Rollback
git reset --hard <commit-hash>

# Restart service
sudo systemctl restart rl-car-racing
```

**Scenario 3: Full System Restore**
```bash
# Restore from backup server
scp -r backup-server:/backups/rl-car-racing/latest/* .

# Reinstall dependencies
pip install -r requirements.txt

# Start fresh
python app.py
```

## Scaling Considerations

### Vertical Scaling (Single Instance)

**Increase Resources**:
- More CPU cores → Faster simulation
- More RAM → Larger replay buffers
- Better GPU → Faster neural network training

**Limits**:
- Single game loop can't parallelize well (sequential simulation)
- Pygame is not multi-threaded
- Diminishing returns beyond 8 cores

### Horizontal Scaling (Multiple Instances)

**Approach 1: Independent Trainers**

Run multiple instances with different:
- Seeds (different tracks)
- Hyperparameters (exploration rates, learning rates)
- Network architectures

**Setup**:
```bash
# Instance 1 (port 5001, seed 42)
python app.py --port 5001 --seed 42 &

# Instance 2 (port 5002, seed 123)
python app.py --port 5002 --seed 123 &

# Instance 3 (port 5003, seed 999)
python app.py --port 5003 --seed 999 &
```

*(Note: Requires modifying app.py to accept CLI arguments)*

**Approach 2: Distributed Training**

Centralized parameter server + multiple workers (requires significant refactoring).

### Load Balancing (Read-Only Instances)

For demonstrations (not training):

**Nginx Configuration**:
```nginx
upstream rl_car_racing {
    server localhost:5001;
    server localhost:5002;
    server localhost:5003;
}

server {
    listen 80;
    location / {
        proxy_pass http://rl_car_racing;
    }
}
```

## Security Best Practices

### Network Security

**Firewall Rules** (UFW on Ubuntu):
```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP (Flask)
sudo ufw allow 5000/tcp

# Enable firewall
sudo ufw enable
```

**HTTPS with Let's Encrypt** (using Nginx):

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com

# Nginx config
server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Authentication

**Add Basic Auth** (Flask):

```python
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash, generate_password_hash

auth = HTTPBasicAuth()

users = {
    "admin": generate_password_hash("secret-password")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')
```

**Environment-Based Credentials**:
```bash
export RL_USERNAME=admin
export RL_PASSWORD=secure-password-here
```

### File System Security

**Restrict Permissions**:
```bash
# Application directory
sudo chown -R rluser:rluser /opt/rl-car-racing
sudo chmod -R 750 /opt/rl-car-racing

# Models directory (sensitive data)
sudo chmod 700 /opt/rl-car-racing/models
```

**Prevent Arbitrary File Access**:

In `load_model` endpoint:
```python
import os

@app.route('/load_model')
def load_model():
    model_file = flask.request.args.get('file', '')
    
    # Sanitize input (prevent path traversal)
    model_file = os.path.basename(model_file)
    
    # Ensure .pkl extension
    if not model_file.endswith('.pkl'):
        return jsonify({'status': 'error', 'message': 'Invalid file type'})
    
    full_path = os.path.join(env.model_save_path, model_file)
    
    # Ensure path is within models directory
    if not os.path.abspath(full_path).startswith(os.path.abspath(env.model_save_path)):
        return jsonify({'status': 'error', 'message': 'Invalid path'})
    
    success = env.load_model(full_path)
    return jsonify({'status': 'loaded' if success else 'failed'})
```

## Troubleshooting

### Common Issues

**Issue**: Video stream shows black screen
**Cause**: Game not started or rendering error
**Solution**:
```bash
# Check game state
curl http://localhost:5000/stats

# Start game
curl http://localhost:5000/start

# Check logs for pygame errors
tail -f training.log
```

---

**Issue**: Training loss is NaN
**Cause**: Numerical instability, gradient explosion
**Solution**:
- Reduce learning rate
- Check for division by zero in reward calculation
- Enable gradient clipping (requires Optax integration)

---

**Issue**: Exploration rate stuck at 1.0
**Cause**: Game loop not running (steps not incrementing)
**Solution**:
```bash
# Verify game is running
curl http://localhost:5000/stats | jq '.exploration_rate'

# Restart game loop
curl http://localhost:5000/stop
curl http://localhost:5000/start
```

---

**Issue**: Model save fails silently
**Cause**: Disk full or permission denied
**Solution**:
```bash
# Check disk space
df -h

# Check permissions
ls -la models/

# Fix permissions
sudo chown -R $USER:$USER models/
```

---

**Issue**: High CPU usage (100% on all cores)
**Cause**: JAX JIT compilation or runaway training loop
**Solution**:
```bash
# Check process
top -p $(pgrep -f "python app.py")

# Disable JIT temporarily
export JAX_DISABLE_JIT=1
python app.py

# Reduce frame rate
# Edit app.py: FPS = 15
```

---

**Issue**: Memory leak (RAM usage grows over time)
**Cause**: Unbounded replay buffer or statistics lists
**Solution**:
- Verify `deque(maxlen=10000)` is set
- Trim statistics lists: `training_losses = training_losses[-100:]`
- Restart service periodically

---

### Debug Mode

**Enable Flask Debug Mode**:
```python
# In app.py, change:
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
```

**Warnings**:
- Never use in production (security risk)
- Auto-reloads on code changes
- Exposes interactive debugger on errors

**JAX Debug Mode**:
```python
# Enable more verbose errors
import jax
jax.config.update('jax_debug_nans', True)  # Stops on NaN
jax.config.update('jax_disable_jit', True)  # Better stack traces
```

### Reporting Issues

When reporting problems, include:

1. **System Info**:
   ```bash
   python --version
   pip list | grep -E "jax|pygame|flask"
   uname -a
   ```

2. **Logs**:
   - Last 100 lines of training.log
   - Flask error output
   - Browser console errors (F12)

3. **Reproducibility**:
   - Steps to reproduce
   - Configuration changes
   - Model checkpoint (if relevant)

4. **Expected vs Actual Behavior**

---

## Summary

This operations guide provides production-ready practices for:
- Deploying across environments (local, server, cloud, container)
- Monitoring training progress and system health
- Managing and backing up model checkpoints
- Tuning performance and scaling resources
- Securing the application
- Troubleshooting common issues

For architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md).
For API reference, see [API_AND_CONFIG.md](API_AND_CONFIG.md).

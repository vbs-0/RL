import flask
from flask import Flask, render_template, Response, jsonify
import pygame
import numpy as np

# Disable JAX JIT compilation to prevent initialization hangs
import os
os.environ['JAX_DISABLE_JIT'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jax import random
import math
import noise
import haiku as hk
from collections import deque
import json
import base64
import io
from PIL import Image
import threading
import time
import pickle

# Import your game code
from test import Car, Track, WorldModel, Actor, BLACK, WHITE, RED, GREEN, BLUE, YELLOW
from rl_system.monitoring import MetricsCollector

app = Flask(__name__)

# Global variables
WIDTH, HEIGHT = 800, 600
FPS = 30
game_state = {
    'running': False,
    'manual_control': True,
    'action': [0.0, 0.0],  # [steering, throttle]
    'obs': None,
    'reward': 0,
    'done': False,
    'frame': None,
    'training_stats': {
        'episode_rewards': [],
        'training_losses': [],
        'exploration_rate': 1.0
    }
}

# Initialize pygame for headless rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class FlaskRLCarEnv:
    def __init__(self):
        self.track = Track()
        # Initialize car at the track's start position
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car = Car(start_x, start_y)
        # Set the initial angle from track
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        
        self.world_model = WorldModel(len(self.car.sensor_angles) + 1, 2)
        self.actor = Actor(32, 2)
        self.key = random.PRNGKey(42)
        self.train_every = 10
        self.step_counter = 0
        self.explore_prob = 1.0
        self.min_explore_prob = 0.1
        self.explore_decay = 0.999
        self.episode_rewards = []
        self.training_losses = []
        self.current_episode_reward = 0
        self.best_progress = 0.0  # Track best progress achieved so far
        self.model_save_path = "models/"
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(log_dir='logs')
        self.metrics_collector.load_metrics_from_disk()
        
        # Episode tracking for metrics
        self.current_episode = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.episode_world_loss = 0.0
        self.episode_actor_loss = 0.0
        self.world_loss_count = 0
        self.actor_loss_count = 0

    def get_observation(self):
        norm_sensors = np.array(self.car.sensor_readings) / self.car.sensor_length
        speed = float(self.car.state[2]) / self.car.max_speed
        obs = np.append(norm_sensors, speed)
        return jnp.array(obs)

    def reset(self):
        # Reset car to track start position
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car.initial_x = start_x
        self.car.initial_y = start_y
        self.car.reset()
        # Set the initial angle from track
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        # Update sensors immediately
        self.car.update_sensors(self.track.track_points, self.track.obstacles)
        self.current_episode_reward = 0
        
        # Reset episode metrics
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.episode_world_loss = 0.0
        self.episode_actor_loss = 0.0
        self.world_loss_count = 0
        self.actor_loss_count = 0
        
        return self.get_observation()

    def step(self, action):
        steering, throttle = action
        on_track = self.car.update([steering, throttle], self.track.track_points, self.track.obstacles)
        next_obs = self.get_observation()
        reward = self.calculate_reward(on_track)
        done = not on_track
        
        self.current_episode_reward += reward
        self.episode_steps += 1

        if self.step_counter % self.train_every == 0:
            self.world_model.add_experience(self.get_observation(), jnp.array(action), next_obs, reward)
            world_loss = self.world_model.train_step()
            self.training_losses.append(world_loss)
            self.episode_world_loss += float(world_loss)
            self.world_loss_count += 1
            
            # Train actor on the latest batch
            if len(self.world_model.buffer) >= self.world_model.batch_size:
                indices = jnp.arange(min(self.world_model.batch_size, len(self.world_model.buffer)))
                batch = [self.world_model.buffer[i] for i in indices]
                latent_batch = jnp.array([self.world_model.encode(item[0]) for item in batch])
                reward_batch = jnp.array([float(item[3][0]) for item in batch])
                actor_loss = self.actor.train_step(latent_batch, reward_batch)
                self.episode_actor_loss += float(actor_loss)
                self.actor_loss_count += 1
            
            # Update global training stats
            game_state['training_stats']['training_losses'] = self.training_losses[-100:]
            game_state['training_stats']['exploration_rate'] = float(self.explore_prob)

        self.step_counter += 1
        self.explore_prob = max(self.min_explore_prob, self.explore_prob * self.explore_decay)
        
        # Check progress and save model only if progress improved
        current_progress = self.car.calculate_track_progress(self.track.track_points)
        if current_progress > self.best_progress:
            self.best_progress = current_progress
            self.save_model()
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            game_state['training_stats']['episode_rewards'] = self.episode_rewards[-100:]
            
            # Calculate steps per second
            episode_duration = time.time() - self.episode_start_time
            steps_per_second = self.episode_steps / max(episode_duration, 0.001)
            
            # Average losses for the episode
            avg_world_loss = (self.episode_world_loss / self.world_loss_count 
                            if self.world_loss_count > 0 else 0.0)
            avg_actor_loss = (self.episode_actor_loss / self.actor_loss_count 
                            if self.actor_loss_count > 0 else 0.0)
            
            # Log episode metrics
            self.metrics_collector.log_episode({
                'episode': self.current_episode,
                'reward': float(self.current_episode_reward),
                'actor_loss': float(avg_actor_loss),
                'world_model_loss': float(avg_world_loss),
                'steps': int(self.episode_steps),
                'steps_per_second': float(steps_per_second),
                'exploration_rate': float(self.explore_prob),
            })
            
            self.current_episode += 1
            
            # Reset current episode reward after episode ends
            self.current_episode_reward = 0
            
        return next_obs, reward, done

    def calculate_reward(self, on_track):
        if not on_track:
            return -10.0

        # Get current speed for reward
        speed_reward = float(self.car.state[2])
        progress = self.car.calculate_track_progress(self.track.track_points)
        progress_reward = 0.0

        # Reward for making progress on the track
        if progress > self.car.last_checkpoint + self.track.checkpoint_interval:
            progress_reward = 5.0
            self.car.last_checkpoint = progress
            current_time = pygame.time.get_ticks()
            time_bonus = 1000 / max(1, current_time - self.car.checkpoint_time)
            progress_reward += time_bonus
            self.car.checkpoint_time = current_time

        # Small reward for staying on track
        survival_reward = 0.1
        total_reward = speed_reward + progress_reward + survival_reward
        return total_reward

    def get_action(self, obs):
        latent = self.world_model.encode(obs)
        explore = random.uniform(self.key, shape=()) < self.explore_prob
        self.key, _ = random.split(self.key)
        action = self.actor.get_action(latent, explore=explore)
        return action

    def render(self):
        screen.fill(BLACK)
        self.track.draw(screen)
        self.car.draw(screen)

        font = pygame.font.SysFont(None, 30)
        x, y, speed, angle = np.array(self.car.state)
        speed_text = font.render(f"Speed: {speed:.1f}", True, WHITE)
        pos_text = font.render(f"Pos: ({int(x)}, {int(y)})", True, WHITE)
        explore_text = font.render(f"Explore: {self.explore_prob:.2f}", True, WHITE)
        progress_text = font.render(f"Progress: {self.car.calculate_track_progress(self.track.track_points):.2f}", True, WHITE)
        screen.blit(speed_text, (10, 10))
        screen.blit(pos_text, (10, 40))
        screen.blit(explore_text, (10, 70))
        screen.blit(progress_text, (10, 100))

        if len(self.world_model.buffer) > 0:
            latent = self.world_model.encode(self.get_observation())
            latent_vis_x = WIDTH - 150
            latent_vis_y = 10
            for i in range(min(16, len(latent))):
                val = max(0, min(255, int(128 + latent[i] * 64)))
                color = (val, 255 - val, val)
                pygame.draw.rect(screen, color, (latent_vis_x + (i % 4) * 20, latent_vis_y + (i // 4) * 20, 15, 15))
            latent_text = font.render("Latent State", True, WHITE)
            screen.blit(latent_text, (WIDTH - 150, 110))

        if game_state['manual_control']:
            control_text = font.render("Manual Control", True, WHITE)
        else:
            control_text = font.render("AI Control", True, WHITE)
        screen.blit(control_text, (WIDTH - 300, HEIGHT - 30))
        
        # Convert the pygame surface to a PIL Image
        data = pygame.image.tostring(screen, 'RGB')
        img = Image.frombytes('RGB', (WIDTH, HEIGHT), data)
        
        # Convert to base64 for streaming
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        frame = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return frame
    
    def save_model(self):
        """Save the model parameters to disk"""
        model_data = {
            'encoder_params': self.world_model.encoder_params,
            'dynamics_params': self.world_model.dynamics_params,
            'decoder_params': self.world_model.decoder_params,
            'reward_params': self.world_model.reward_params,
            'actor_params': self.actor.actor_params,
            'explore_prob': self.explore_prob,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses
        }
        
        filename = f"{self.model_save_path}model_ep{len(self.episode_rewards)}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load model parameters from disk"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.world_model.encoder_params = model_data['encoder_params']
            self.world_model.dynamics_params = model_data['dynamics_params']
            self.world_model.decoder_params = model_data['decoder_params']
            self.world_model.reward_params = model_data['reward_params']
            self.actor.actor_params = model_data['actor_params']
            self.explore_prob = model_data['explore_prob']
            self.episode_rewards = model_data['episode_rewards']
            self.training_losses = model_data['training_losses']
            
            # Update global training stats
            game_state['training_stats']['episode_rewards'] = self.episode_rewards[-100:]
            game_state['training_stats']['training_losses'] = self.training_losses[-100:]
            game_state['training_stats']['exploration_rate'] = float(self.explore_prob)
            
            print(f"Model loaded from {filename}")
            return True
        else:
            print(f"Model file {filename} not found")
            return False

# Initialize environment
env = FlaskRLCarEnv()
game_state['obs'] = env.reset()

def game_loop():
    """Main game loop that runs in a separate thread"""
    while True:
        if game_state['running']:
            if game_state['manual_control']:
                action = game_state['action']
            else:
                action = env.get_action(game_state['obs'])
            
            next_obs, reward, done = env.step(action)
            
            game_state['obs'] = next_obs
            game_state['reward'] = reward
            game_state['done'] = done
            
            if done:
                game_state['obs'] = env.reset()
            
            # Render and update frame
            game_state['frame'] = env.render()
            
            # Control frame rate
            clock.tick(FPS)
        else:
            time.sleep(0.1)  # Sleep when not running to reduce CPU usage

# Start game loop in a separate thread
game_thread = threading.Thread(target=game_loop, daemon=True)
game_thread.start()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            if game_state['frame']:
                # Decode the base64 encoded JPEG to raw bytes
                jpeg_bytes = base64.b64decode(game_state['frame'])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
            time.sleep(1/FPS)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start_game():
    """Start the game"""
    game_state['running'] = True
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop_game():
    """Stop the game"""
    game_state['running'] = False
    return jsonify({'status': 'stopped'})

@app.route('/reset')
def reset_game():
    """Reset the game"""
    game_state['obs'] = env.reset()
    return jsonify({'status': 'reset'})

@app.route('/toggle_control')
def toggle_control():
    """Toggle between manual and AI control"""
    game_state['manual_control'] = not game_state['manual_control']
    return jsonify({'status': 'toggled', 'manual_control': game_state['manual_control']})

@app.route('/control', methods=['POST'])
def control():
    """Receive control inputs from the client"""
    data = flask.request.get_json()
    if 'steering' in data and 'throttle' in data:
        game_state['action'] = [float(data['steering']), float(data['throttle'])]
    return jsonify({'status': 'ok'})

@app.route('/stats')
def get_stats():
    """Return current training statistics with monitoring metrics"""
    stats = game_state['training_stats'].copy()
    
    # Add metrics from the collector
    metrics_history = env.metrics_collector.get_metrics_history()
    aggregated_stats = env.metrics_collector.get_aggregated_stats()
    
    # Enrich stats with additional metrics
    stats.update({
        'metrics_history': metrics_history,
        'aggregated_stats': aggregated_stats,
        'steps_per_second': (
            aggregated_stats.get('total_steps', 0) / 
            max(aggregated_stats.get('uptime_seconds', 1), 1)
        ),
        'uptime_seconds': aggregated_stats.get('uptime_seconds', 0),
    })
    
    return jsonify(stats)

@app.route('/save_model')
def save_model():
    """Save the current model"""
    env.save_model()
    return jsonify({'status': 'saved'})

@app.route('/load_model')
def load_model():
    """Load a model"""
    model_file = flask.request.args.get('file', '')
    if model_file:
        success = env.load_model(model_file)
        return jsonify({'status': 'loaded' if success else 'failed'})
    else:
        return jsonify({'status': 'error', 'message': 'No model file specified'})

@app.route('/list_models')
def list_models():
    """List all saved models"""
    models = []
    if os.path.exists(env.model_save_path):
        models = [f for f in os.listdir(env.model_save_path) if f.endswith('.pkl')]
    return jsonify({'models': models})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
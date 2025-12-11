"""Flask web application for the RL car system."""

import flask
from flask import Flask, render_template, Response, jsonify
import pygame
import numpy as np
import jax.numpy as jnp
import os
import io
import base64
from PIL import Image
import threading
import time
import pickle
import signal
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_system.training import ContinuousTrainer
from rl_system.config import config

# Disable JAX JIT compilation to prevent initialization hangs
os.environ['JAX_DISABLE_JIT'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

app = Flask(__name__)

# Global variables for web interface
game_state = {
    'running': False,
    'manual_control': True,
    'action': [0.0, 0.0],  # [steering, throttle]
    'frame': None,
    'current_model': None,
    'available_models': []
}

# Global trainer instance
trainer = None

def initialize_trainer():
    """Initialize the global trainer instance."""
    global trainer
    
    # Load default configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'default.yaml')
    
    # Initialize trainer in headless mode for web interface
    trainer = ContinuousTrainer(config_path=config_path, headless=True)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("Received shutdown signal, stopping trainer...")
        if trainer:
            trainer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Trainer initialized for web interface")

def game_loop():
    """Main game loop that runs in a separate thread."""
    global trainer
    
    print("Starting game loop thread...")
    
    while True:
        if game_state['running'] and trainer and trainer.running:
            # Update current action from web interface
            trainer.manual_control = game_state['manual_control']
            trainer.current_action = game_state['action']
            
            # Run a single step
            obs = trainer.env.get_observation()
            
            # Get action based on mode
            if trainer.manual_control:
                action = jnp.array(game_state['action'])
            else:
                action = trainer.get_action(obs)
            
            # Step environment
            next_obs, reward, done = trainer.env.step(np.array(action))
            
            # Train on the experience
            losses = trainer.train_step(obs, action, next_obs, reward)
            
            # Check for model saving
            trainer.check_and_save_model()
            
            # Handle episode completion
            if done:
                trainer.episode_rewards.append(trainer.current_episode_reward)
                trainer.current_episode_reward = 0
                
                # Auto-save every few episodes
                if trainer.episode_counter % trainer.save_interval == 0:
                    trainer.save_model()
            
            # Render frame for web streaming
            trainer.env.render(trainer.screen)
            game_state['frame'] = encode_frame(trainer.screen)
            
            # Update available models
            game_state['available_models'] = trainer.list_models()
            
            # Control frame rate
            trainer.clock.tick(trainer.fps)
        else:
            time.sleep(0.1)  # Sleep when not running

def encode_frame(screen):
    """Encode pygame surface to base64 JPEG for web streaming."""
    try:
        # Convert the pygame surface to a PIL Image
        data = pygame.image.tostring(screen, 'RGB')
        width, height = screen.get_size()
        img = Image.frombytes('RGB', (width, height), data)
        
        # Convert to base64 for streaming
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        frame = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return frame
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        while True:
            if game_state['frame']:
                # Decode the base64 encoded JPEG to raw bytes
                jpeg_bytes = base64.b64decode(game_state['frame'])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')
            time.sleep(1/30)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_game():
    """Start the training/game."""
    global trainer
    if not trainer:
        initialize_trainer()
    
    game_state['running'] = True
    trainer.running = True
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop_game():
    """Stop the training/game."""
    global trainer
    game_state['running'] = False
    if trainer:
        trainer.stop()
    return jsonify({'status': 'stopped'})

@app.route('/reset')
def reset_game():
    """Reset the environment."""
    global trainer
    if trainer:
        trainer.env.reset()
        trainer.current_episode_reward = 0
    return jsonify({'status': 'reset'})

@app.route('/toggle_control')
def toggle_control():
    """Toggle between manual and AI control."""
    game_state['manual_control'] = not game_state['manual_control']
    return jsonify({'status': 'toggled', 'manual_control': game_state['manual_control']})

@app.route('/control', methods=['POST'])
def control():
    """Receive control inputs from the client."""
    data = flask.request.get_json()
    if 'steering' in data and 'throttle' in data:
        game_state['action'] = [float(data['steering']), float(data['throttle'])]
    return jsonify({'status': 'ok'})

@app.route('/stats')
def get_stats():
    """Return current training statistics."""
    global trainer
    if trainer:
        return jsonify(trainer.get_training_stats())
    else:
        return jsonify({
            'running': False,
            'episode': 0,
            'step': 0,
            'exploration_rate': 1.0,
            'episode_rewards': [],
            'training_losses': [],
            'buffer_size': 0
        })

@app.route('/save_model')
def save_model():
    """Save the current model."""
    global trainer
    if trainer:
        trainer.save_model()
        return jsonify({'status': 'saved'})
    return jsonify({'status': 'error', 'message': 'Trainer not initialized'})

@app.route('/load_model')
def load_model():
    """Load a model."""
    global trainer
    model_file = flask.request.args.get('file', '')
    if model_file and trainer:
        model_path = os.path.join(trainer.model_save_path, model_file)
        success = trainer.load_model(model_path)
        return jsonify({'status': 'loaded' if success else 'failed'})
    return jsonify({'status': 'error', 'message': 'No model file specified or trainer not initialized'})

@app.route('/list_models')
def list_models():
    """List all saved models."""
    global trainer
    if trainer:
        models = trainer.list_models()
        return jsonify({'models': models})
    return jsonify({'models': []})

@app.route('/set_config', methods=['POST'])
def set_config():
    """Update configuration dynamically."""
    global trainer
    data = flask.request.get_json()
    
    if data and trainer:
        try:
            # Update configuration values
            for key, value in data.items():
                config.set(key, value)
                print(f"Updated config: {key} = {value}")
            
            return jsonify({'status': 'updated', 'message': 'Configuration updated successfully'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    return jsonify({'status': 'error', 'message': 'No data provided or trainer not initialized'})

def main():
    """Main entry point for the Flask app."""
    # Initialize trainer
    initialize_trainer()
    
    # Start game loop in a separate thread
    game_thread = threading.Thread(target=game_loop, daemon=True)
    game_thread.start()
    
    # Get configuration for server settings
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"Starting Flask server on {host}:{port}")
    print("Use the web interface to control training")
    print("Available endpoints:")
    print("  /           - Main interface")
    print("  /start      - Start training")
    print("  /stop       - Stop training")
    print("  /stats      - Get training statistics")
    print("  /save_model - Save current model")
    print("  /list_models - List available models")
    
    # Run Flask app
    app.run(host=host, port=port, threaded=True)

if __name__ == '__main__':
    main()
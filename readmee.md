# RL Car Game - Advanced Reinforcement Learning and Simulation

![Project Logo](https://avatars.githubusercontent.com/u/137281646?s=200&v=4)

## Overview

This project is an advanced Reinforcement Learning (RL) based car game simulation. It integrates state-of-the-art techniques and libraries to simulate a realistic driving environment, train an RL agent using a DreamerV3-inspired world model, and provide real-time interaction via a web interface. The project demonstrates multiple advanced programming techniques including:
- **Flask Web Server**: Serve and control the game remotely.
- **Pygame Rendering**: Real-time simulation and visualization.
- **Reinforcement Learning**: World model and actor for decision making.
- **JAX and Haiku**: Accelerated computations for neural network evaluations.
- **Resource Management**: Efficient saving and loading of trained models.
- **Threading**: Concurrent execution of the game loop and web server.

## Project Structure

```
newgame/
├── .qodo
│   └── history.sqlite
├── app.py
├── car_game.py
├── game.py
├── models/
│   ├── model_ep0.pkl
│   ├── model_ep1.pkl
│   ├── model_ep2.pkl
│   ├── model_ep3.pkl
│   ├── model_ep4.pkl
│   ├── model_ep6.pkl
│   └── model_ep81.pkl
├── templates/
│   └── index.html
└── test.py
```

- **app.py**:  
  The main entry point for the Flask web application. It integrates the Pygame-based simulation into a headless rendering mode, streaming game frames as JPEG images through a video feed endpoint (`/video_feed`) and exposing various control endpoints (start, stop, reset, toggle manual/AI control, save/load models, etc.).

- **car_game.py**:  
  Contains a standalone Pygame-based implementation of the car game. This script illustrates a more manual implementation with key-based control and uses JAX for accelerated computations.

- **game.py**:  
  Implements an RL-enabled version of the car game using a DreamerV3-inspired world model. It includes:
  - **Car Class**: Handles physics, sensor updates, collision detection, and track progress.
  - **Track Class**: Generates dynamic tracks using Perlin noise with obstacles.
  - **WorldModel Class**: A simplified world model using Haiku and JAX for predicting future states, rewards, and encoding observations.
  - **Actor Class**: Policy network for selecting actions based on latent representations.
  - **RLCarEnv Class**: Integrates simulation steps, environment resets, reward calculation, and interaction with the RL models.
  
- **templates/index.html**:  
  Provides the web interface for remote control and monitoring of the game. It includes:
  - A video stream section showing live game frames.
  - Controls to start, stop, and reset the game; toggle between manual and AI control.
  - On-screen controls and keyboard events for manual control.
  - Model management functionalities (save, list, and load models).
  - Dashboard displaying training statistics (exploration rate, episode rewards, training losses) with real-time charts.

## Dependencies

The project leverages the following libraries and frameworks:
- **Flask**: For web server and API endpoints.
- **Pygame**: For game rendering and handling real-time simulation.
- **JAX**: Accelerated numerical computations and automatic differentiation.
- **Haiku**: Neural network library built on top of JAX.
- **NumPy**: Array and numerical operations.
- **noise**: To generate Perlin noise for organic track creation.
- **PIL (Pillow)**: For image processing and converting Pygame surfaces to images.
- **Threading**: To run the game loop concurrently with the Flask server.
- **Pickle**: For model saving and loading.

## How It Works

### Game Initialization and Loop
- **FlaskRLCarEnv / RLCarEnv**:  
  Sets up the environment, initializes the car at the start position, updates sensor readings, and manages the RL components (world model and actor).  
- **Game Loop**:  
  Runs in a separate thread when enabled. The loop:
  - Retrieves either manual control input (from client POST requests) or selects actions via the RL agent.
  - Steps the simulation forward and calculates rewards.
  - Triggers RL training steps periodically using experience replay.
  - Renders game frames that are streamed over HTTP.

### RL Component Details
- **World Model**:  
  Encodes observations into latent space and predicts future states and rewards. It uses JAX for performance and is updated using experiences stored in a replay buffer.
- **Actor**:  
  A neural network that decides the next action based on the latent representation. It includes an exploration mechanism that decays over time.
- **Training and Experience Replay**:  
  Experiences are stored in a fixed-size buffer and periodically used to train the world model. Training losses and rewards are tracked and updated on the client-side dashboard.

### Resource Management
- **Model Saving/Loading**:  
  The game periodically saves improved models (when track progress improves) to the `models/` directory. Endpoints allow for saving and loading these pickle files during remote training.
- **Logging and Statistics**:  
  Training statistics (episode rewards, training losses, and exploration rates) are maintained globally and served to the client for real-time visualization and analysis.

## Control and Endpoints

The Flask app exposes multiple endpoints to manage and interact with the game:
- `GET /`: Serves the main HTML page.
- `GET /video_feed`: Streams the game frames as an MJPEG feed.
- `GET /start`: Starts the game loop.
- `GET /stop`: Stops the game loop.
- `GET /reset`: Resets the game/simulation.
- `GET /toggle_control`: Toggles between manual control and AI control.
- `POST /control`: Receives manual control inputs (steering and throttle).
- `GET /stats`: Provides current training statistics.
- `GET /save_model`: Saves the current model parameters to disk.
- `GET /load_model`: Loads a specified model from disk.
- `GET /list_models`: Lists all saved model files in the `models/` directory.

## Setup and Running

1. **Install Dependencies**:  
   Use pip to install all required libraries:
   ```
   pip install flask pygame jax jaxlib haiku numpy pillow noise
   ```
2. **Run the Game with Web Interface**:  
   Execute `app.py` to launch both the Flask server and start the game loop:
   ```
   python app.py
   ```
3. **Access the Game**:  
   Open your browser and navigate to `http://localhost:5000`. Use the on-screen buttons and controls for interaction.

## Conclusion

This project is a demonstration of integrating advanced RL techniques using JAX and Haiku with a real-time simulation environment provided by Pygame and Flask. Its design ensures:
- Efficient resource management.
- Real-time visualization and control.
- Detailed logging and monitoring of training progress.
- Flexibility to switch between manual and AI-controlled gameplay.

Feel free to explore, modify, and extend the project to experiment with new RL algorithms and simulation improvements!
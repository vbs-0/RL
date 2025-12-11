
import pygame
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad
import math
import noise
import haiku as hk
from collections import deque
import optax
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Car Game with DreamerV3")
clock = pygame.time.Clock()
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Car:
    def __init__(self, x, y):
        self.initial_x = x
        self.initial_y = y
        self.state = jnp.array([x, y, 0.0, 0.0])  # [x, y, speed, angle]
        self.max_speed = 5.0
        self.size = 20
        self.sensor_length = 150
        self.sensor_angles = jnp.array([-45, -22.5, 0, 22.5, 45])
        self.sensor_readings = jnp.ones(len(self.sensor_angles)) * self.sensor_length
        self.last_checkpoint = 0
        self.checkpoint_time = pygame.time.get_ticks()
        self.prev_position = (x, y)

    def reset(self):
        self.state = jnp.array([self.initial_x, self.initial_y, 0.0, 0.0])
        self.sensor_readings = jnp.ones(len(self.sensor_angles)) * self.sensor_length
        self.last_checkpoint = 0
        self.checkpoint_time = pygame.time.get_ticks()
        self.prev_position = (self.initial_x, self.initial_y)

    def update(self, action, track_points, obstacles):
        x, y, speed, angle = np.array(self.state)
        steering, throttle = action

        # Improved speed calculation with better acceleration and deceleration
        speed = np.clip(speed + throttle * 0.2 - 0.01 * speed, 0, self.max_speed)
        angle = (angle + steering * 5.0) % 360  # Increased steering sensitivity

        self.prev_position = (x, y)
        x = x + np.cos(np.radians(angle)) * speed
        y = y - np.sin(np.radians(angle)) * speed

        if x < 0:
            x = 0
        elif x > WIDTH:
            x = WIDTH
        if y < 0:
            y = 0
        elif y > HEIGHT:
            y = HEIGHT

        self.state = jnp.array([x, y, speed, angle])
        self.update_sensors(track_points, obstacles)
        on_track = self.is_on_track(track_points)
        return on_track

    def update_sensors(self, track_points, obstacles):
        x, y, _, angle = np.array(self.state)
        readings = []

        for sensor_angle in self.sensor_angles:
            ray_angle = angle + sensor_angle
            ray_x = x + self.sensor_length * np.cos(np.radians(ray_angle))
            ray_y = y - self.sensor_length * np.sin(np.radians(ray_angle))
            min_distance = self.sensor_length

            for i in range(len(track_points) - 1):
                p1 = track_points[i]
                p2 = track_points[i + 1]
                intersection = self.line_intersection((x, y), (ray_x, ray_y), p1, p2)
                if intersection:
                    ix, iy = intersection
                    dist = np.sqrt((x - ix)**2 + (y - iy)**2)
                    min_distance = min(min_distance, dist)

            for obs in obstacles:
                cx, cy = obs[0], obs[1]
                radius = 10
                dx = cx - x
                dy = cy - y
                dirx = np.cos(np.radians(ray_angle))
                diry = -np.sin(np.radians(ray_angle))
                t = dx * dirx + dy * diry
                px = x + t * dirx
                py = y + t * diry
                dist_to_center = np.sqrt((px - cx)**2 + (py - cy)**2)
                if dist_to_center <= radius and 0 <= t <= self.sensor_length:
                    adjusted_dist = t - np.sqrt(radius**2 - dist_to_center**2)
                    min_distance = min(min_distance, max(0, adjusted_dist))

            readings.append(min_distance)

        self.sensor_readings = jnp.array(readings)

    def line_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-6:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)

        return None

    def is_on_track(self, track_points):
        x, y, _, _ = np.array(self.state)
        track_width = 40

        for i in range(len(track_points) - 1):
            p1 = track_points[i]
            p2 = track_points[i + 1]
            v1x = p2[0] - p1[0]
            v1y = p2[1] - p1[1]
            v2x = x - p1[0]
            v2y = y - p1[1]
            segment_length = np.sqrt(v1x**2 + v1y**2)
            projection = (v2x * v1x + v2y * v1y) / segment_length

            if 0 <= projection <= segment_length:
                px = v1y / segment_length
                py = -v1x / segment_length
                distance = abs(v2x * px + v2y * py)
                if distance <= track_width:
                    return True

        return False

    def calculate_track_progress(self, track_points):
        x, y, _, _ = np.array(self.state)
        min_dist = float('inf')
        nearest_segment = 0

        for i in range(len(track_points) - 1):
            p1 = track_points[i]
            p2 = track_points[i + 1]
            v1x = p2[0] - p1[0]
            v1y = p2[1] - p1[1]
            v2x = x - p1[0]
            v2y = y - p1[1]
            segment_length = np.sqrt(v1x**2 + v1y**2)
            projection = (v2x * v1x + v2y * v1y) / segment_length

            if 0 <= projection <= segment_length:
                px = v1y / segment_length
                py = -v1x / segment_length
                distance = abs(v2x * px + v2y * py)
                if distance < min_dist:
                    min_dist = distance
                    nearest_segment = i + projection / segment_length

        return nearest_segment / (len(track_points) - 1)

    def draw(self, screen):
        x, y, _, angle = np.array(self.state)
        points = [
            (int(x + self.size * math.cos(math.radians(angle))),
            int(y - self.size * math.sin(math.radians(angle)))),
            (int(x + self.size * 0.7 * math.cos(math.radians(angle + 135))),
            int(y - self.size * 0.7 * math.sin(math.radians(angle + 135)))),
            (int(x + self.size * 0.7 * math.cos(math.radians(angle - 135))),
            int(y - self.size * 0.7 * math.sin(math.radians(angle - 135))))
        ]
        pygame.draw.polygon(screen, RED, points)

        for i, sensor_angle in enumerate(self.sensor_angles):
            ray_angle = angle + sensor_angle
            ray_length = float(self.sensor_readings[i])
            ray_x = x + ray_length * math.cos(math.radians(ray_angle))
            ray_y = y - ray_length * math.sin(math.radians(ray_angle))
            color_intensity = int(255 * ray_length / self.sensor_length)
            sensor_color = (255 - color_intensity, color_intensity, 0)
            pygame.draw.line(screen, sensor_color, (int(x), int(y)), (int(ray_x), int(ray_y)), 2)

class Track:
    def __init__(self, seed=42):
        self.seed = seed
        self.track_points = self.generate_track()
        self.obstacles = self.generate_obstacles()
        self.checkpoint_interval = 0.1
        self.start_position = self.get_start_position()

    '''def generate_track(self):
        points = []
        num_points = 100
        radius = min(WIDTH, HEIGHT) * 0.35
        center_x, center_y = WIDTH // 2, HEIGHT // 2

        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            noise_value = noise.pnoise1(i * 0.1, base=self.seed)
            r = radius * (1 + 0.3 * noise_value)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            points.append((int(x), int(y)))

        points.append(points[0])
        return points'''
    def generate_track(self):
        # Simple circular track generation without noise
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        radius = min(WIDTH, HEIGHT) * 0.35
        num_points = 50  # Fewer points for a simpler track
        points = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = center_x + int(radius * math.cos(theta))
            y = center_y + int(radius * math.sin(theta))
            points.append((x, y))
        # Close the track by appending the first point at the end
        points.append(points[0])
        return points
    
    def generate_obstacles(self):
        obstacles = []
        for i in range(5):
            angle = 2 * math.pi * i / 5
            radius = min(WIDTH, HEIGHT) * 0.2
            noise_val = noise.pnoise1(i * 0.5, base=self.seed + 1)
            x = WIDTH // 2 + radius * math.cos(angle) * (1 + noise_val)
            y = HEIGHT // 2 + radius * math.sin(angle) * (1 + noise_val)
            obstacles.append((int(x), int(y)))
        return obstacles

    def get_start_position(self):
        # Get a position on the track for the car to start
        p1 = self.track_points[0]
        p2 = self.track_points[1]
        
        # Calculate direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Normalize and get perpendicular vector
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            # Calculate angle from direction vector
            angle = math.degrees(math.atan2(-dy, dx))
            
            # Return starting position and angle
            return (p1[0], p1[1], angle)
        
        return (p1[0], p1[1], 0)  # Default if calculation fails

    def draw(self, screen):
    # Draw the center line of the track
        pygame.draw.lines(screen, WHITE, False, self.track_points, 3)
    
        track_width = 40
    
    # Create more detailed track boundaries
    # 1. Create and draw complete inside and outside track boundaries
        inside_points = []
        outside_points = []

        for i in range(len(self.track_points)):
            p1 = self.track_points[i]
            p2 = self.track_points[(i + 1) % len(self.track_points)]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)

            if length > 0:
                nx = dy / length
                ny = -dx / length

                inside_points.append((int(p1[0] + nx * track_width), int(p1[1] + ny * track_width)))
            outside_points.append((int(p1[0] - nx * track_width), int(p1[1] - ny * track_width)))

        # Connect the last point to the first to close the track visually
        pygame.draw.lines(screen, (150, 150, 150), True, inside_points, 2)
        pygame.draw.lines(screen, (150, 150, 150), True, outside_points, 2)
    
        # 2. Add track markers (stripes) along the track for visual detail
        stripe_interval = 5  # Every 5 points add a stripe across the track
        for i in range(0, len(self.track_points), stripe_interval):
            if i < len(self.track_points) - 1:
                p1 = self.track_points[i]
                p2 = self.track_points[(i + 1) % len(self.track_points)]

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx**2 + dy**2)

                if length > 0:
                    nx = dy / length
                    ny = -dx / length

                    inside_point = (int(p1[0] + nx * track_width), int(p1[1] + ny * track_width))
                    outside_point = (int(p1[0] - nx * track_width), int(p1[1] - ny * track_width))

                    # Draw a stripe across the track
                    pygame.draw.line(screen, (200, 200, 200), inside_point, outside_point, 1)

        # 3. Add track edge details (small dots at intervals)
        dot_interval = 3  # Every 3 points add a dot on each edge
        for i in range(0, len(inside_points), dot_interval):
            pygame.draw.circle(screen, (255, 255, 255), inside_points[i], 2)
        pygame.draw.circle(screen, (255, 255, 255), outside_points[i], 2)
        
        # 4. Add colored sections to make the track more interesting
        # Divide the track into sections and color them differently
        num_sections = 6
        section_size = len(self.track_points) // num_sections
        for s in range(num_sections):
            start_idx = s * section_size
            end_idx = (s + 1) * section_size
            
            # Create section boundary points
            section_color = (50 + s * 30, 50, 150 - s * 20)  # Varying colors
            for i in range(start_idx, end_idx, 2):  # Every other point
                if i < len(self.track_points) - 1:
                    p1 = self.track_points[i]
                    p2 = self.track_points[(i + 1) % len(self.track_points)]
                    
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = math.sqrt(dx**2 + dy**2)
                    
                    if length > 0:
                        nx = dy / length
                        ny = -dx / length
                        
                        # Draw subtle colored section indicators on the inside of the track
                        pygame.draw.circle(screen, section_color, 
                                         (int(p1[0] + nx * track_width * 0.8), 
                                          int(p1[1] + ny * track_width * 0.8)), 3)
    
    # Draw obstacles
        for obs in self.obstacles:
            # Add a glowing effect to obstacles
            for radius in range(15, 5, -3):
                alpha = (15 - radius) * 17  # Decreasing alpha for outer circles
                color = (min(255, YELLOW[0] - alpha), min(255, YELLOW[1] - alpha), 0)
                pygame.draw.circle(screen, color, (int(obs[0]), int(obs[1])), radius)

            # Main obstacle
            pygame.draw.circle(screen, YELLOW, (int(obs[0]), int(obs[1])), 10)

    # Draw a more visible start/finish line
        start = self.track_points[0]
        p1 = self.track_points[1]
        dx = p1[0] - start[0]
        dy = p1[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = dy / length * track_width
            ny = -dx / length * track_width
            
            # Draw a checkered start/finish line
            start_point1 = (int(start[0] + nx), int(start[1] + ny))
            start_point2 = (int(start[0] - nx), int(start[1] - ny))
            pygame.draw.line(screen, GREEN, start_point1, start_point2, 5)
        
            # Add checkered pattern
            num_checks = 8
            for i in range(num_checks):
                check_start = i / num_checks
                check_end = (i + 1) / num_checks
                if i % 2 == 0:  # Alternate black and white
                    p1 = (int(start_point1[0] * (1 - check_start) + start_point2[0] * check_start),
                          int(start_point1[1] * (1 - check_start) + start_point2[1] * check_start))
                    p2 = (int(start_point1[0] * (1 - check_end) + start_point2[0] * check_end),
                          int(start_point1[1] * (1 - check_end) + start_point2[1] * check_end))
                    pygame.draw.line(screen, BLACK, p1, p2, 5)

class WorldModel:
    def __init__(self, obs_size, action_size, seed=42):
        self.obs_size = obs_size
        self.action_size = action_size
        self.latent_size = 32
        self.key = random.PRNGKey(seed)
        
        # Define world model network
        def encoder_fn(obs):
            net = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(self.latent_size)
            ])
            return net(obs)
        
        def dynamics_fn(latent, action):
            net = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(self.latent_size)
            ])
            x = jnp.concatenate([latent, action], axis=-1)
            return net(x)
        
        def decoder_fn(latent):
            net = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(self.obs_size)
            ])
            return net(latent)
        
        def reward_fn(latent):
            net = hk.Sequential([
                hk.Linear(32), jax.nn.relu,
                hk.Linear(1)
            ])
            return net(latent)
        
        # Transform into pure functions
        self.encoder = hk.transform(encoder_fn)
        self.dynamics = hk.transform(dynamics_fn)
        self.decoder = hk.transform(decoder_fn)
        self.reward = hk.transform(reward_fn)
        
        # Initialize parameters
        dummy_obs = jnp.zeros((1, self.obs_size))
        dummy_latent = jnp.zeros((1, self.latent_size))
        dummy_action = jnp.zeros((1, self.action_size))
        
        self.key, subkey1, subkey2, subkey3, subkey4 = random.split(self.key, 5)
        self.encoder_params = self.encoder.init(subkey1, dummy_obs)
        self.dynamics_params = self.dynamics.init(subkey2, dummy_latent, dummy_action)
        self.decoder_params = self.decoder.init(subkey3, dummy_latent)
        self.reward_params = self.reward.init(subkey4, dummy_latent)
        
        # Initialize optimizers for each network
        self.encoder_optimizer = optax.adam(learning_rate=1e-3)
        self.dynamics_optimizer = optax.adam(learning_rate=1e-3)
        self.decoder_optimizer = optax.adam(learning_rate=1e-3)
        self.reward_optimizer = optax.adam(learning_rate=1e-3)
        
        # Initialize optimizer states
        self.encoder_opt_state = self.encoder_optimizer.init(self.encoder_params)
        self.dynamics_opt_state = self.dynamics_optimizer.init(self.dynamics_params)
        self.decoder_opt_state = self.decoder_optimizer.init(self.decoder_params)
        self.reward_opt_state = self.reward_optimizer.init(self.reward_params)
        
        # Buffer for experience replay
        self.buffer = deque(maxlen=10000)
        self.batch_size = 32
    
    def encode(self, obs):
        # Convert observation to batch (add batch dimension)
        obs_batch = jnp.expand_dims(obs, axis=0)
        return self.encoder.apply(self.encoder_params, None, obs_batch)[0]
    
    def predict_next(self, latent, action):
        # Add batch dimensions
        latent_batch = jnp.expand_dims(latent, axis=0)
        action_batch = jnp.expand_dims(action, axis=0)
        
        next_latent = self.dynamics.apply(self.dynamics_params, None, latent_batch, action_batch)[0]
        predicted_reward = float(self.reward.apply(self.reward_params, None, latent_batch)[0][0])
        
        return next_latent, predicted_reward
    
    def decode(self, latent):
        # Add batch dimension
        latent_batch = jnp.expand_dims(latent, axis=0)
        return self.decoder.apply(self.decoder_params, None, latent_batch)[0]
    
    def add_experience(self, obs, action, next_obs, reward):
        self.buffer.append((obs, action, next_obs, reward))
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0  # Not enough data
        
        # Sample batch
        indices = random.randint(self.key, (self.batch_size,), 0, len(self.buffer))
        self.key, _ = random.split(self.key)
        
        batch = [self.buffer[int(i)] for i in indices]
        obs_batch = jnp.array([np.array(item[0]) for item in batch])
        action_batch = jnp.array([np.array(item[1]) for item in batch])
        next_obs_batch = jnp.array([np.array(item[2]) for item in batch])
        # Extract scalar rewards
        reward_batch = jnp.array([float(item[3]) for item in batch])
        
        # Define loss function for all networks
        def compute_loss(encoder_params, dynamics_params, decoder_params, reward_params):
            # Get latent representations
            latent_batch = self.encoder.apply(encoder_params, None, obs_batch)
            
            # Predict next latent and reward
            next_latent_batch = self.dynamics.apply(dynamics_params, None, latent_batch, action_batch)
            predicted_reward = self.reward.apply(reward_params, None, latent_batch)  # Shape: (batch_size, 1)
            
            # Decode to observations
            decoded_obs = self.decoder.apply(decoder_params, None, latent_batch)
            decoded_next_obs = self.decoder.apply(decoder_params, None, next_latent_batch)
            
            # Calculate losses
            reconstruction_loss = jnp.mean((decoded_obs - obs_batch) ** 2)
            next_reconstruction_loss = jnp.mean((decoded_next_obs - next_obs_batch) ** 2)
            reward_loss = jnp.mean((predicted_reward.squeeze() - reward_batch) ** 2)
            
            total_loss = reconstruction_loss + next_reconstruction_loss + reward_loss
            return total_loss, (reconstruction_loss, next_reconstruction_loss, reward_loss)
        
        # Compute gradients for all parameters
        # Compute gradients for all parameters
        grad_fn = grad(compute_loss, argnums=(0, 1, 2, 3), has_aux=True)
        grads_and_aux = grad_fn(self.encoder_params, self.dynamics_params, self.decoder_params, self.reward_params)
        (encoder_grads, dynamics_grads, decoder_grads, reward_grads), loss_components = grads_and_aux
        total_loss = sum(loss_components)
        # Update encoder
        encoder_updates, self.encoder_opt_state = self.encoder_optimizer.update(
            encoder_grads, self.encoder_opt_state
        )
        self.encoder_params = optax.apply_updates(self.encoder_params, encoder_updates)
        
        # Update dynamics
        dynamics_updates, self.dynamics_opt_state = self.dynamics_optimizer.update(
            dynamics_grads, self.dynamics_opt_state
        )
        self.dynamics_params = optax.apply_updates(self.dynamics_params, dynamics_updates)
        
        # Update decoder
        decoder_updates, self.decoder_opt_state = self.decoder_optimizer.update(
            decoder_grads, self.decoder_opt_state
        )
        self.decoder_params = optax.apply_updates(self.decoder_params, decoder_updates)
        
        # Update reward
        reward_updates, self.reward_opt_state = self.reward_optimizer.update(
            reward_grads, self.reward_opt_state
        )
        self.reward_params = optax.apply_updates(self.reward_params, reward_updates)
        
        # Debug: Print actual loss value
        actual_loss = float(total_loss)
        if actual_loss > 0:
            print(f"WorldModel Loss: {actual_loss:.6f}, Reconstruction: {float(loss_components[0]):.6f}")
        
        return actual_loss

class Actor:
    def __init__(self, latent_size, action_size, seed=42):
        self.latent_size = latent_size
        self.action_size = action_size
        self.key = random.PRNGKey(seed)
        
        # Define actor network
        def actor_fn(latent):
            net = hk.Sequential([
                hk.Linear(64), jax.nn.relu,
                hk.Linear(32), jax.nn.relu,
                hk.Linear(action_size), jax.nn.tanh
            ])
            return net(latent)
        
        # Transform into pure function
        self.actor = hk.transform(actor_fn)
        
        # Initialize parameters
        dummy_latent = jnp.zeros((1, self.latent_size))
        self.key, subkey = random.split(self.key)
        self.actor_params = self.actor.init(subkey, dummy_latent)
        
        # Initialize optimizer
        self.actor_optimizer = optax.adam(learning_rate=1e-3)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)
    
    def get_action(self, latent, explore=False):
        # Add batch dimension
        latent_batch = jnp.expand_dims(latent, axis=0)
        
        # Get action from policy
        action = self.actor.apply(self.actor_params, None, latent_batch)[0]
        
        # Add exploration noise if needed
        if explore:
            self.key, subkey = random.split(self.key)
            noise = random.normal(subkey, shape=action.shape) * 0.2
            action = jnp.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def train_step(self, latent_batch, reward_batch):
        """Train actor to maximize expected rewards using policy gradient"""
        def compute_actor_loss(actor_params):
            # Get actions from policy
            actions = self.actor.apply(actor_params, None, latent_batch)
            
            # Policy gradient loss: -mean(reward * log_prob)
            # Approximated by: -mean(reward * action_output)
            actor_loss = -jnp.mean(reward_batch.reshape(-1, 1) * actions)
            return actor_loss
        
        # Compute gradient
        actor_grads = grad(compute_actor_loss)(self.actor_params)
        
        # Update parameters
        actor_updates, self.actor_opt_state = self.actor_optimizer.update(
            actor_grads, self.actor_opt_state
        )
        self.actor_params = optax.apply_updates(self.actor_params, actor_updates)
        
        return float(compute_actor_loss(self.actor_params))

class RLCarEnv:
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
        return self.get_observation()

    def step(self, action):
        # Ensure action is properly scaled for the car
        steering, throttle = action
        on_track = self.car.update([steering, throttle], self.track.track_points, self.track.obstacles)
        next_obs = self.get_observation()
        reward = self.calculate_reward(on_track)
        done = not on_track

        if self.step_counter % self.train_every == 0 and len(self.world_model.buffer) < self.world_model.buffer.maxlen:
            self.world_model.add_experience(self.get_observation(), jnp.array(action), next_obs, jnp.array([reward]))
            loss = self.world_model.train_step()
            self.training_losses.append(loss)

        self.step_counter += 1
        self.explore_prob = max(self.min_explore_prob, self.explore_prob * self.explore_decay)
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

    def render(self, screen):
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

def main():
    env = RLCarEnv()
    obs = env.reset()
    running = True
    manual_control = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    manual_control = not manual_control
                elif event.key == pygame.K_r:
                    obs = env.reset()

        if manual_control:
            keys = pygame.key.get_pressed()
            steering = 0.0
            throttle = 0.0
            if keys[pygame.K_LEFT]:
                steering = -1.0
            if keys[pygame.K_RIGHT]:
                steering = 1.0
            if keys[pygame.K_UP]:
                throttle = 1.0
            if keys[pygame.K_DOWN]:
                throttle = -1.0
            action = jnp.array([steering, throttle])
        else:
            action = env.get_action(obs)

        next_obs, reward, done = env.step(action)
        obs = next_obs

        if done:
            obs = env.reset()

        env.render(screen)
        font = pygame.font.SysFont(None, 30)
        if manual_control:
            control_text = font.render("Manual Control (Space to toggle)", True, WHITE)
        else:
            control_text = font.render("AI Control (Space to toggle)", True, WHITE)
        screen.blit(control_text, (WIDTH - 300, HEIGHT - 30))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()

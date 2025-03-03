import pygame
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random
import math
import noise
import haiku as hk
from collections import deque
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
        self.x = x
        self.y = y
        self.vel = 0
        self.angle = 0


    def reset(self):
    # Reset position, velocity, angle, etc., to initial values
        self.x = self.initial_x
        self.y = self.initial_y
        self.vel = 0
        self.angle = 0
        self.state = jnp.array([self.x, self.y, self.vel, self.angle])
          # Update state
    def update(self, action, track_points, obstacles):
    # Convert to numpy for PyGame compatibility
        x, y, speed, angle = np.array(self.state)
        steering, throttle = action
    
    # Update speed and angle
        speed = np.clip(speed + throttle * 0.1 - 0.01, 0, self.max_speed)
        angle = (angle + steering * 3.0) % 360
    
    # Update position
        self.prev_position = (x, y)
        x = x + np.cos(np.radians(angle)) * speed
        y = y - np.sin(np.radians(angle)) * speed
    
    # Boundary checks
        if x < 0:
            x = 0
        elif x > WIDTH: 
            x = WIDTH
        if y < 0:
            y = 0
        elif y > HEIGHT:
            y = HEIGHT
    
    # Update state
        self.state = jnp.array([x, y, speed, angle])
    
    # Update sensors
        self.update_sensors(track_points, obstacles)
    
    # Check if car is on track
        on_track = self.is_on_track(track_points)
        return on_track

    def update_sensors(self, track_points, obstacles):
        x, y, _, angle = np.array(self.state)
        readings = []
        
        for sensor_angle in self.sensor_angles:
            ray_angle = angle + sensor_angle
            ray_x = x + self.sensor_length * np.cos(np.radians(ray_angle))
            ray_y = y - self.sensor_length * np.sin(np.radians(ray_angle))
            
            # Start with max length
            min_distance = self.sensor_length
            
            # Check track boundaries (simplified as line segments)
            for i in range(len(track_points) - 1):
                p1 = track_points[i]
                p2 = track_points[i + 1]
                
                # Check intersection
                intersection = self.line_intersection((x, y), (ray_x, ray_y), p1, p2)
                if intersection:
                    ix, iy = intersection
                    dist = np.sqrt((x - ix)**2 + (y - iy)**2)
                    min_distance = min(min_distance, dist)
            
            # Check obstacles
            for obs in obstacles:
                # Simple circle-line intersection test
                cx, cy = obs[0], obs[1]
                radius = 10  # Obstacle radius
                
                # Vector from line start to circle center
                dx = cx - x
                dy = cy - y
                
                # Direction vector of the line
                dirx = np.cos(np.radians(ray_angle))
                diry = -np.sin(np.radians(ray_angle))
                
                # Project circle center onto line
                t = dx * dirx + dy * diry
                
                # Closest point on line to circle center
                px = x + t * dirx
                py = y + t * diry
                
                # Distance from closest point to circle center
                dist_to_center = np.sqrt((px - cx)**2 + (py - cy)**2)
                
                if dist_to_center <= radius and 0 <= t <= self.sensor_length:
                    # Adjust distance to account for obstacle radius
                    adjusted_dist = t - np.sqrt(radius**2 - dist_to_center**2)
                    min_distance = min(min_distance, max(0, adjusted_dist))
            
            readings.append(min_distance)
        
        self.sensor_readings = jnp.array(readings)

    def line_intersection(self, p1, p2, p3, p4):
        # Calculate line intersection using determinants
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-6:
            return None  # Lines are parallel
        
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
        
        return None

    def is_on_track(self, track_points):
        x, y, _, _ = np.array(self.state)
        
        # Define track width
        track_width = 40
        
        # Check if point is within track_width of any track segment
        for i in range(len(track_points) - 1):
            p1 = track_points[i]
            p2 = track_points[i + 1]
            
            # Vector from p1 to p2
            v1x = p2[0] - p1[0]
            v1y = p2[1] - p1[1]
            
            # Vector from p1 to point
            v2x = x - p1[0]
            v2y = y - p1[1]
            
            # Length of segment
            segment_length = np.sqrt(v1x**2 + v1y**2)
            
            # Calculate projection
            projection = (v2x * v1x + v2y * v1y) / segment_length
            
            # Check if projection is on segment
            if 0 <= projection <= segment_length:
                # Normalized perpendicular vector
                px = v1y / segment_length
                py = -v1x / segment_length
                
                # Distance to line
                distance = abs(v2x * px + v2y * py)
                
                if distance <= track_width:
                    return True
        
        return False

    def calculate_track_progress(self, track_points):
        x, y, _, _ = np.array(self.state)
        
        # Find nearest track segment
        min_dist = float('inf')
        nearest_segment = 0
        
        for i in range(len(track_points) - 1):
            p1 = track_points[i]
            p2 = track_points[i + 1]
            
            # Vector from p1 to p2
            v1x = p2[0] - p1[0]
            v1y = p2[1] - p1[1]
            
            # Vector from p1 to point
            v2x = x - p1[0]
            v2y = y - p1[1]
            
            # Length of segment
            segment_length = np.sqrt(v1x**2 + v1y**2)
            
            # Calculate projection
            projection = (v2x * v1x + v2y * v1y) / segment_length
            
            # Check if projection is on segment
            if 0 <= projection <= segment_length:
                # Normalized perpendicular vector
                px = v1y / segment_length
                py = -v1x / segment_length
                
                # Distance to line
                distance = abs(v2x * px + v2y * py)
                
                if distance < min_dist:
                    min_dist = distance
                    nearest_segment = i + projection / segment_length
        
        return nearest_segment / (len(track_points) - 1)

    def draw(self, screen):
        x, y, _, angle = np.array(self.state)
    
        # Draw car body as triangle
        points = [
    (int(x + self.size * math.cos(math.radians(angle))),
     int(y - self.size * math.sin(math.radians(angle)))),
    (int(x + self.size * 0.7 * math.cos(math.radians(angle + 135))),
     int(y - self.size * 0.7 * math.sin(math.radians(angle + 135)))),
    (int(x + self.size * 0.7 * math.cos(math.radians(angle - 135))),
     int(y - self.size * 0.7 * math.sin(math.radians(angle - 135)))),
]
        pygame.draw.polygon(screen, RED, points)
    
        # Draw sensors
        for i, sensor_angle in enumerate(self.sensor_angles):
            ray_angle = angle + sensor_angle
            ray_length = float(self.sensor_readings[i])
            ray_x = x + ray_length * math.cos(math.radians(ray_angle))
            ray_y = y - ray_length * math.sin(math.radians(ray_angle))
        
            # Color based on distance (red to green)
            color_intensity = int(255 * ray_length / self.sensor_length)
            sensor_color = (255 - color_intensity, color_intensity, 0)
        
            # Convert coordinates to integers before passing to pygame.draw.line
            pygame.draw.line(screen, sensor_color, (int(x), int(y)), (int(ray_x), int(ray_y)), 2)

# Advanced Track with Perlin Noise
class Track:
    def __init__(self, seed=42):
        self.seed = seed
        self.track_points = self.generate_track()
        self.obstacles = self.generate_obstacles()
        self.checkpoint_interval = 0.1  # Progress interval for checkpoints

    def generate_track(self):
        points = []
        # Create a closed loop track with Perlin noise
        num_points = 100
        radius = min(WIDTH, HEIGHT) * 0.35
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            # Vary radius with Perlin noise
            noise_value = noise.pnoise1(i * 0.1, base=self.seed)
            r = radius * (1 + 0.3 * noise_value)
            
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            points.append((int(x), int(y)))
        
        # Make the track loop by connecting last point to first
        points.append(points[0])
        return points

    def generate_obstacles(self):
        obstacles = []
        # Place obstacles around the track
        for i in range(5):
            # Use noise to place obstacles
            angle = 2 * math.pi * i / 5
            radius = min(WIDTH, HEIGHT) * 0.2
            noise_val = noise.pnoise1(i * 0.5, base=self.seed + 1)
            
            x = WIDTH // 2 + radius * math.cos(angle) * (1 + noise_val)
            y = HEIGHT // 2 + radius * math.sin(angle) * (1 + noise_val)
            obstacles.append((int(x), int(y)))
        
        return obstacles

    def draw(self, screen):
        # Draw track
        pygame.draw.lines(screen, WHITE, False, self.track_points, 3)
        
        # Draw track outline (width guide)
        track_width = 40
        inside_points = []
        outside_points = []
        
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i + 1]
            
            # Vector from p1 to p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            
            # Normalize and rotate 90 degrees for perpendicular
            if length > 0:
                nx = dy / length
                ny = -dx / length
                
                # Inside and outside points
                inside_points.append((int(p1[0] + nx * track_width), int(p1[1] + ny * track_width)))
                outside_points.append((int(p1[0] - nx * track_width), int(p1[1] - ny * track_width)))
        
        # Draw track bounds
        pygame.draw.lines(screen, (100, 100, 100), False, inside_points, 1)
        pygame.draw.lines(screen, (100, 100, 100), False, outside_points, 1)
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.circle(screen, YELLOW, (int(obs[0]), int(obs[1])), 10)
        
        # Draw start/finish line
        start = self.track_points[0]
        p1 = self.track_points[1]
        
        # Vector perpendicular to track direction
        dx = p1[0] - start[0]
        dy = p1[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)
        
        if length > 0:
            nx = dy / length * track_width
            ny = -dx / length * track_width
            
            # Draw start/finish line
            pygame.draw.line(screen, GREEN, 
                             (int(start[0] + nx), int(start[1] + ny)),
                             (int(start[0] - nx), int(start[1] - ny)), 5)

# DreamerV3 (simplified world model implementation)
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
        predicted_reward = float(self.reward.apply(self.reward_params, None, latent_batch)[0])
        
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
        obs_batch = jnp.array([item[0] for item in batch])
        action_batch = jnp.array([item[1] for item in batch])
        next_obs_batch = jnp.array([item[2] for item in batch])
        reward_batch = jnp.array([item[3] for item in batch])
        
        # Get latent representations
        latent_batch = self.encoder.apply(self.encoder_params, None, obs_batch)
        
        # Predict next latent and reward
        next_latent_batch = self.dynamics.apply(self.dynamics_params, None, latent_batch, action_batch)
        predicted_reward = self.reward.apply(self.reward_params, None, latent_batch)
        
        # Decode to observations
        decoded_obs = self.decoder.apply(self.decoder_params, None, latent_batch)
        decoded_next_obs = self.decoder.apply(self.decoder_params, None, next_latent_batch)
        
        # Calculate losses (simplified)
        reconstruction_loss = jnp.mean((decoded_obs - obs_batch) ** 2)
        next_reconstruction_loss = jnp.mean((decoded_next_obs - next_obs_batch) ** 2)
        reward_loss = jnp.mean((predicted_reward - reward_batch) ** 2)
        
        total_loss = reconstruction_loss + next_reconstruction_loss + reward_loss
        
        # In a real implementation, we would update parameters with optimizer here
        # For simplicity, we're just returning the loss
        return float(total_loss)

# Actor model for policy
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

# Reinforcement Learning environment
class RLCarEnv:
    def __init__(self):
        self.car = Car(WIDTH // 2, HEIGHT // 2)
        self.track = Track()
        
        # State representation: sensor readings + speed
        obs_size = len(self.car.sensor_angles) + 1  # sensors + speed
        action_size = 2  # [steering, throttle]
        
        # Initialize world model and actor
        self.world_model = WorldModel(obs_size, action_size)
        self.actor = Actor(32, action_size)  # 32 is latent size from world model
        
        # Initialize random key for exploration
        self.key = random.PRNGKey(42)  # You can choose any seed
        
        # Training parameters
        self.train_every = 10
        self.step_counter = 0
        self.explore_prob = 1.0
        self.min_explore_prob = 0.1
        self.explore_decay = 0.999
        
        # Performance tracking
        self.episode_rewards = []
        self.training_losses = []
    
    def get_observation(self):
        # Normalize sensor readings
        norm_sensors = np.array(self.car.sensor_readings) / self.car.sensor_length
        
        # Normalize speed
        speed = float(self.car.state[2]) / self.car.max_speed
        
        # Combine into observation vector
        obs = np.append(norm_sensors, speed)
        return jnp.array(obs)
    
    def reset(self):
    # Set the car's position to the first point of the track
        self.car.x, self.car.y = self.track.track_points[0]
        self.car.angle = 0  # Reset angle to face the track direction
        self.car.reset()  # This calls the Car's reset method
        return self.get_observation()
    
    def step(self, action):
        # Convert action from [-1, 1] to appropriate ranges
        steering = float(action[0])  # -1 to 1
        throttle = (float(action[1]) + 1) / 2  # 0 to 1
        
        # Update car
        on_track = self.car.update([steering, throttle], 
                                    self.track.track_points,
                                    self.track.obstacles)
        
        # Get new observation
        next_obs = self.get_observation()
        
        # Calculate reward
        reward = self.calculate_reward(on_track)
        
        # Check if episode is done
        done = not on_track
        
        # Add to experience buffer for training
        if self.step_counter % self.train_every == 0:
            self.world_model.add_experience(
                self.get_observation(),
                jnp.array(action),
                next_obs,
                jnp.array([reward])
            )
            loss = self.world_model.train_step()
            self.training_losses.append(loss)
        
        self.step_counter += 1
        
        # Decay exploration probability
        self.explore_prob = max(self.min_explore_prob, 
                                self.explore_prob * self.explore_decay)
        
        return next_obs, reward, done
    
    def calculate_reward(self, on_track):
        if not on_track:
            return -10.0  # Penalty for going off track
        
        # Reward for speed
        speed_reward = float(self.car.state[2])
        
        # Reward for progress on track
        progress = self.car.calculate_track_progress(self.track.track_points)
        progress_reward = 0.0
        
        # Check if we've passed a checkpoint
        if progress > self.car.last_checkpoint + self.track.checkpoint_interval:
            # Reward for reaching next checkpoint
            progress_reward = 5.0
            self.car.last_checkpoint = progress
            
            # Additional time bonus
            current_time = pygame.time.get_ticks()
            time_bonus = 1000 / max(1, current_time - self.car.checkpoint_time)
            progress_reward += time_bonus
            self.car.checkpoint_time = current_time
        
        # Small reward for staying on track
        survival_reward = 0.1
        
        # Combine rewards
        total_reward = speed_reward + progress_reward + survival_reward
        return total_reward
    
    def get_action(self, obs):
        # Encode observation to latent state
        latent = self.world_model.encode(obs)
        
        # Decide whether to explore
        explore = random.uniform(self.key, shape=()) < self.explore_prob
        self.key, _ = random.split(self.key)
        
        # Get action from policy
        action = self.actor.get_action(latent, explore=explore)
        return action
    
    def render(self, screen):
        # Clear screen
        screen.fill(BLACK)
        
        # Draw track and car
        self.track.draw(screen)
        self.car.draw(screen)
        
        # Draw UI elements
        font = pygame.font.SysFont(None, 30)
        x, y, speed, angle = np.array(self.car.state)
        
        # Speed and position
        speed_text = font.render(f"Speed: {speed:.1f}", True, WHITE)
        pos_text = font.render(f"Pos: ({int(x)}, {int(y)})", True, WHITE)
        
        # Training info
        explore_text = font.render(f"Explore: {self.explore_prob:.2f}", True, WHITE)
        progress_text = font.render(f"Progress: {self.car.calculate_track_progress(self.track.track_points):.2f}", True, WHITE)
        
        # Display texts
        screen.blit(speed_text, (10, 10))
        screen.blit(pos_text, (10, 40))
        screen.blit(explore_text, (10, 70))
        screen.blit(progress_text, (10, 100))
        
        # Visualize latent space (simplified)
        if len(self.world_model.buffer) > 0:
            latent = self.world_model.encode(self.get_observation())
            
            # Show simplified latent visualization as small colored squares
            latent_vis_x = WIDTH - 150
            latent_vis_y = 10
            for i in range(min(16, len(latent))):
                val = max(0, min(255, int(128 + latent[i] * 64)))
                color = (val, 255 - val, val)
                pygame.draw.rect(screen, color, 
                                (latent_vis_x + (i % 4) * 20,
                                 latent_vis_y + (i // 4) * 20, 15, 15))
            
            # Label
            latent_text = font.render("Latent State", True, WHITE)
            screen.blit(latent_text, (WIDTH - 150, 110))

# Main function
def main():
    env = RLCarEnv()
    obs = env.reset()
    
    running = True
    manual_control = True  # Start with manual control
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle between manual and AI control
                    manual_control = not manual_control
                elif event.key == pygame.K_r:
                    # Reset environment
                    obs = env.reset()
        
        if manual_control:
            # Manual control with keyboard
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
            # AI control
            action = env.get_action(obs)
        
        # Step environment
        next_obs, reward, done = env.step(action)
        obs = next_obs
        
        if done:
            # Reset if done
            obs = env.reset()
        
        # Render
        env.render(screen)
        
        # Add control mode indicator
        font = pygame.font.SysFont(None, 30)
        if manual_control:
            control_text = font.render("Manual Control (Space to toggle)", True, WHITE)
        else:
            control_text = font.render("AI Control (Space to toggle)", True, WHITE)
        screen.blit(control_text, (WIDTH - 300, HEIGHT - 30))
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()

"""Environment module containing Car, Track, and RLCarEnv classes."""

import pygame
import numpy as np
import jax
import jax.numpy as jnp
import math
import noise
from typing import Tuple, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_system.config import config


class Car:
    """Car agent with physics, sensors, and track awareness."""
    
    def __init__(self, x: float, y: float):
        """Initialize car at position (x, y)."""
        self.initial_x = x
        self.initial_y = y
        self.state = jnp.array([x, y, 0.0, 0.0])  # [x, y, speed, angle]
        
        # Load configuration
        self.max_speed = config.get('car.max_speed', 5.0)
        self.size = config.get('car.size', 20)
        self.sensor_length = config.get('car.sensor_length', 150)
        self.sensor_angles = jnp.array(config.get('car.sensor_angles', [-45, -22.5, 0, 22.5, 45]))
        self.sensor_readings = jnp.ones(len(self.sensor_angles)) * self.sensor_length
        
        # Track progress tracking
        self.last_checkpoint = 0
        self.checkpoint_time = pygame.time.get_ticks()
        self.prev_position = (x, y)

    def reset(self):
        """Reset car to initial state."""
        self.state = jnp.array([self.initial_x, self.initial_y, 0.0, 0.0])
        self.sensor_readings = jnp.ones(len(self.sensor_angles)) * self.sensor_length
        self.last_checkpoint = 0
        self.checkpoint_time = pygame.time.get_ticks()
        self.prev_position = (self.initial_x, self.initial_y)

    def update(self, action: List[float], track_points: List[Tuple[float, float]], 
               obstacles: List[Tuple[float, float]]) -> bool:
        """Update car state based on action and environment."""
        x, y, speed, angle = np.array(self.state)
        steering, throttle = action

        # Speed calculation with acceleration and deceleration
        speed = np.clip(speed + throttle * 0.2 - 0.01 * speed, 0, self.max_speed)
        angle = (angle + steering * 5.0) % 360

        self.prev_position = (x, y)
        x = x + np.cos(np.radians(angle)) * speed
        y = y - np.sin(np.radians(angle)) * speed

        # Boundary constraints
        width = config.get('environment.width', 800)
        height = config.get('environment.height', 600)
        
        x = np.clip(x, 0, width)
        y = np.clip(y, 0, height)

        self.state = jnp.array([x, y, speed, angle])
        self.update_sensors(track_points, obstacles)
        on_track = self.is_on_track(track_points)
        return on_track

    def update_sensors(self, track_points: List[Tuple[float, float]], 
                      obstacles: List[Tuple[float, float]]):
        """Update sensor readings based on track and obstacles."""
        x, y, _, angle = np.array(self.state)
        readings = []

        for sensor_angle in self.sensor_angles:
            ray_angle = angle + sensor_angle
            ray_x = x + self.sensor_length * np.cos(np.radians(ray_angle))
            ray_y = y - self.sensor_length * np.sin(np.radians(ray_angle))
            min_distance = self.sensor_length

            # Check track intersections
            for i in range(len(track_points) - 1):
                p1 = track_points[i]
                p2 = track_points[i + 1]
                intersection = self.line_intersection((x, y), (ray_x, ray_y), p1, p2)
                if intersection:
                    ix, iy = intersection
                    dist = np.sqrt((x - ix)**2 + (y - iy)**2)
                    min_distance = min(min_distance, dist)

            # Check obstacle intersections
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

    def line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float],
                         p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Calculate intersection of two line segments."""
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

    def is_on_track(self, track_points: List[Tuple[float, float]]) -> bool:
        """Check if car is on the track."""
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

    def calculate_track_progress(self, track_points: List[Tuple[float, float]]) -> float:
        """Calculate progress along the track (0.0 to 1.0)."""
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
        """Draw the car and sensors on the screen."""
        x, y, _, angle = np.array(self.state)
        points = [
            (int(x + self.size * math.cos(math.radians(angle))),
             int(y - self.size * math.sin(math.radians(angle)))),
            (int(x + self.size * 0.7 * math.cos(math.radians(angle + 135))),
             int(y - self.size * 0.7 * math.sin(math.radians(angle + 135)))),
            (int(x + self.size * 0.7 * math.cos(math.radians(angle - 135))),
             int(y - self.size * 0.7 * math.sin(math.radians(angle - 135))))
        ]
        
        colors = config.get('colors', {})
        red = tuple(colors.get('red', [255, 0, 0]))
        pygame.draw.polygon(screen, red, points)

        # Draw sensors
        for i, sensor_angle in enumerate(self.sensor_angles):
            ray_angle = angle + sensor_angle
            ray_length = float(self.sensor_readings[i])
            ray_x = x + ray_length * math.cos(math.radians(ray_angle))
            ray_y = y - ray_length * math.sin(math.radians(ray_angle))
            color_intensity = int(255 * ray_length / self.sensor_length)
            sensor_color = (255 - color_intensity, color_intensity, 0)
            pygame.draw.line(screen, sensor_color, (int(x), int(y)), (int(ray_x), int(ray_y)), 2)


class Track:
    """Procedurally generated racing track."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize track with optional seed."""
        if seed is None:
            seed = config.get('environment.track_seed', 42)
        self.seed = seed
        self.track_points = self.generate_track()
        self.obstacles = self.generate_obstacles()
        self.checkpoint_interval = config.get('environment.track_checkpoint_interval', 0.1)
        self.start_position = self.get_start_position()

    def generate_track(self) -> List[Tuple[int, int]]:
        """Generate a circular track with optional noise."""
        width = config.get('environment.width', 800)
        height = config.get('environment.height', 600)
        
        # Simple circular track generation
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) * 0.35
        num_points = 50  # Fewer points for a simpler track
        points = []
        
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = center_x + int(radius * math.cos(theta))
            y = center_y + int(radius * math.sin(theta))
            points.append((x, y))
        
        # Close the track
        points.append(points[0])
        return points
    
    def generate_obstacles(self) -> List[Tuple[int, int]]:
        """Generate obstacles around the track."""
        width = config.get('environment.width', 800)
        height = config.get('environment.height', 600)
        
        obstacles = []
        for i in range(5):
            angle = 2 * math.pi * i / 5
            radius = min(width, height) * 0.2
            noise_val = noise.pnoise1(i * 0.5, base=self.seed + 1)
            x = width // 2 + radius * math.cos(angle) * (1 + noise_val)
            y = height // 2 + radius * math.sin(angle) * (1 + noise_val)
            obstacles.append((int(x), int(y)))
        return obstacles

    def get_start_position(self) -> Tuple[float, float, float]:
        """Get starting position and angle for the car."""
        p1 = self.track_points[0]
        p2 = self.track_points[1]
        
        # Calculate direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Normalize and get angle
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            angle = math.degrees(math.atan2(-dy, dx))
            return (p1[0], p1[1], angle)
        
        return (p1[0], p1[1], 0)

    def draw(self, screen):
        """Draw the track with enhanced visuals."""
        colors = config.get('colors', {})
        white = tuple(colors.get('white', [255, 255, 255]))
        black = tuple(colors.get('black', [0, 0, 0]))
        yellow = tuple(colors.get('yellow', [255, 255, 0]))
        green = tuple(colors.get('green', [0, 255, 0]))
        
        # Draw the center line of the track
        pygame.draw.lines(screen, white, False, self.track_points, 3)
        
        track_width = 40
        
        # Create track boundaries
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

        # Draw track boundaries
        pygame.draw.lines(screen, (150, 150, 150), True, inside_points, 2)
        pygame.draw.lines(screen, (150, 150, 150), True, outside_points, 2)
        
        # Draw start/finish line
        start = self.track_points[0]
        p1 = self.track_points[1]
        dx = p1[0] - start[0]
        dy = p1[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = dy / length * track_width
            ny = -dx / length * track_width
            
            start_point1 = (int(start[0] + nx), int(start[1] + ny))
            start_point2 = (int(start[0] - nx), int(start[1] - ny))
            pygame.draw.line(screen, green, start_point1, start_point2, 5)

        # Draw obstacles
        for obs in self.obstacles:
            # Add glowing effect
            for radius in range(15, 5, -3):
                alpha = (15 - radius) * 17
                color = (min(255, yellow[0] - alpha), min(255, yellow[1] - alpha), 0)
                pygame.draw.circle(screen, color, (int(obs[0]), int(obs[1])), radius)

            pygame.draw.circle(screen, yellow, (int(obs[0]), int(obs[1])), 10)


class RLCarEnv:
    """Reinforcement Learning environment for the car racing task."""
    
    def __init__(self):
        """Initialize the RL environment."""
        self.track = Track()
        
        # Initialize car at the track's start position
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car = Car(start_x, start_y)
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        
        # Training parameters
        self.train_every = config.get('training.train_every', 10)
        self.step_counter = 0
        self.explore_prob = config.get('training.exploration_initial', 1.0)
        self.min_explore_prob = config.get('training.exploration_min', 0.1)
        self.explore_decay = config.get('training.exploration_decay', 0.999)
        self.episode_rewards = []
        self.training_losses = []
        self.current_episode_reward = 0
        self.best_progress = 0.0

    def get_observation(self) -> jnp.ndarray:
        """Get normalized observation from sensors and speed."""
        norm_sensors = np.array(self.car.sensor_readings) / self.car.sensor_length
        speed = float(self.car.state[2]) / self.car.max_speed
        obs = np.append(norm_sensors, speed)
        return jnp.array(obs)

    def reset(self) -> jnp.ndarray:
        """Reset environment to initial state."""
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car.initial_x = start_x
        self.car.initial_y = start_y
        self.car.reset()
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        self.car.update_sensors(self.track.track_points, self.track.obstacles)
        self.current_episode_reward = 0
        return self.get_observation()

    def step(self, action: List[float]) -> Tuple[jnp.ndarray, float, bool]:
        """Execute one environment step."""
        steering, throttle = action
        on_track = self.car.update([steering, throttle], self.track.track_points, self.track.obstacles)
        next_obs = self.get_observation()
        reward = self.calculate_reward(on_track)
        done = not on_track

        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

        return next_obs, reward, done

    def calculate_reward(self, on_track: bool) -> float:
        """Calculate reward based on performance."""
        if not on_track:
            return -10.0

        # Speed reward
        speed_reward = float(self.car.state[2])
        
        # Progress reward
        progress = self.car.calculate_track_progress(self.track.track_points)
        progress_reward = 0.0

        if progress > self.car.last_checkpoint + self.track.checkpoint_interval:
            progress_reward = 5.0
            self.car.last_checkpoint = progress
            current_time = pygame.time.get_ticks()
            time_bonus = 1000 / max(1, current_time - self.car.checkpoint_time)
            progress_reward += time_bonus
            self.car.checkpoint_time = current_time

        # Survival reward
        survival_reward = 0.1
        total_reward = speed_reward + progress_reward + survival_reward
        return total_reward

    def render(self, screen):
        """Render the environment to a pygame surface."""
        colors = config.get('colors', {})
        black = tuple(colors.get('black', [0, 0, 0]))
        white = tuple(colors.get('white', [255, 255, 255]))
        
        screen.fill(black)
        self.track.draw(screen)
        self.car.draw(screen)

        font = pygame.font.SysFont(None, 30)
        x, y, speed, angle = np.array(self.car.state)
        speed_text = font.render(f"Speed: {speed:.1f}", True, white)
        pos_text = font.render(f"Pos: ({int(x)}, {int(y)})", True, white)
        explore_text = font.render(f"Explore: {self.explore_prob:.2f}", True, white)
        progress_text = font.render(f"Progress: {self.car.calculate_track_progress(self.track.track_points):.2f}", True, white)
        screen.blit(speed_text, (10, 10))
        screen.blit(pos_text, (10, 40))
        screen.blit(explore_text, (10, 70))
        screen.blit(progress_text, (10, 100))

        # Visualize latent state if available
        if hasattr(self, 'world_model') and len(self.world_model.buffer) > 0:
            try:
                latent = self.world_model.encode(self.get_observation())
                latent_vis_x = config.get('environment.width', 800) - 150
                latent_vis_y = 10
                for i in range(min(16, len(latent))):
                    val = max(0, min(255, int(128 + latent[i] * 64)))
                    color = (val, 255 - val, val)
                    pygame.draw.rect(screen, color, (latent_vis_x + (i % 4) * 20, latent_vis_y + (i // 4) * 20, 15, 15))
                latent_text = font.render("Latent State", True, white)
                screen.blit(latent_text, (config.get('environment.width', 800) - 150, 110))
            except:
                pass  # Skip latent visualization if world model not available
#!/usr/bin/env python3
"""
Manual Drive Example - Keyboard-Controlled Episode

This script demonstrates manual control of the car using keyboard inputs.
The agent can be switched between manual and AI control on the fly.

Controls:
    Arrow Keys: Steer (LEFT/RIGHT) and accelerate/brake (UP/DOWN)
    SPACE: Toggle between manual and AI control
    R: Reset episode
    Q/ESC: Quit

Usage:
    python examples/manual_drive.py [--seed SEED] [--fps FPS]
"""
import argparse
import sys
import os

# Add parent directory to path to import test module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['JAX_DISABLE_JIT'] = '1'

import pygame
import numpy as np
import jax.numpy as jnp
from test import Car, Track, WorldModel, Actor, WIDTH, HEIGHT, BLACK, WHITE


class ManualDriveEnv:
    """Environment for manual driving with keyboard controls."""
    
    def __init__(self, seed=42):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Manual Drive Example - RL Car Game")
        self.clock = pygame.time.Clock()
        
        # Create environment
        self.track = Track(seed=seed)
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car = Car(start_x, start_y)
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        
        # Initialize models for AI mode
        obs_size = len(self.car.sensor_angles) + 1
        action_size = 2
        self.world_model = WorldModel(obs_size, action_size, seed=seed)
        self.actor = Actor(32, action_size, seed=seed)
        
        # Training parameters
        self.train_every = 10
        self.step_counter = 0
        self.explore_prob = 0.3  # Lower exploration for demo
        
        # Statistics
        self.episode_reward = 0.0
        self.manual_control = True
    
    def get_observation(self):
        """Get current observation."""
        norm_sensors = np.array(self.car.sensor_readings) / self.car.sensor_length
        speed = float(self.car.state[2]) / self.car.max_speed
        obs = np.append(norm_sensors, speed)
        return jnp.array(obs)
    
    def reset(self):
        """Reset environment."""
        start_x, start_y, start_angle = self.track.get_start_position()
        self.car.state = jnp.array([start_x, start_y, 0.0, start_angle])
        self.car.reset()
        self.car.update_sensors(self.track.track_points, self.track.obstacles)
        self.episode_reward = 0.0
        return self.get_observation()
    
    def calculate_reward(self, on_track):
        """Calculate reward."""
        if not on_track:
            return -10.0
        
        speed_reward = float(self.car.state[2])
        progress = self.car.calculate_track_progress(self.track.track_points)
        progress_reward = 0.0
        
        if progress > self.car.last_checkpoint + self.track.checkpoint_interval:
            progress_reward = 5.0
            self.car.last_checkpoint = progress
            current_time = pygame.time.get_ticks()
            time_bonus = 1000 / max(1, current_time - self.car.checkpoint_time)
            progress_reward += time_bonus
            self.car.checkpoint_time = current_time
        
        survival_reward = 0.1
        return speed_reward + progress_reward + survival_reward
    
    def step(self, action):
        """Execute one step."""
        steering = float(action[0])
        throttle = (float(action[1]) + 1) / 2
        
        on_track = self.car.update([steering, throttle],
                                   self.track.track_points,
                                   self.track.obstacles)
        
        next_obs = self.get_observation()
        reward = self.calculate_reward(on_track)
        done = not on_track
        
        self.episode_reward += reward
        
        # Optional: train in background
        if self.step_counter % self.train_every == 0:
            self.world_model.add_experience(
                self.get_observation(),
                jnp.array(action),
                next_obs,
                reward
            )
            self.world_model.train_step()
        
        self.step_counter += 1
        
        return next_obs, reward, done
    
    def get_action_from_policy(self, obs):
        """Get AI action."""
        from jax import random
        latent = self.world_model.encode(obs)
        explore = random.uniform(self.world_model.key, shape=()) < self.explore_prob
        self.world_model.key, _ = random.split(self.world_model.key)
        action = self.actor.get_action(latent, explore=explore)
        return action
    
    def get_manual_action(self):
        """Get action from keyboard."""
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
        
        return jnp.array([steering, throttle])
    
    def render(self):
        """Render the environment."""
        self.screen.fill(BLACK)
        
        # Draw track and car
        self.track.draw(self.screen)
        self.car.draw(self.screen)
        
        # Draw UI
        font = pygame.font.SysFont(None, 30)
        x, y, speed, angle = np.array(self.car.state)
        
        # Stats
        speed_text = font.render(f"Speed: {speed:.1f}", True, WHITE)
        pos_text = font.render(f"Pos: ({int(x)}, {int(y)})", True, WHITE)
        progress = self.car.calculate_track_progress(self.track.track_points)
        progress_text = font.render(f"Progress: {progress:.2f}", True, WHITE)
        reward_text = font.render(f"Reward: {self.episode_reward:.1f}", True, WHITE)
        
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(pos_text, (10, 40))
        self.screen.blit(progress_text, (10, 70))
        self.screen.blit(reward_text, (10, 100))
        
        # Control mode
        if self.manual_control:
            control_text = font.render("MANUAL CONTROL", True, (0, 255, 0))
        else:
            control_text = font.render("AI CONTROL", True, (255, 255, 0))
        self.screen.blit(control_text, (WIDTH - 200, HEIGHT - 30))
        
        # Instructions
        small_font = pygame.font.SysFont(None, 20)
        instructions = [
            "Controls:",
            "Arrows: Drive",
            "SPACE: Toggle Mode",
            "R: Reset",
            "Q/ESC: Quit"
        ]
        for i, text in enumerate(instructions):
            inst_text = small_font.render(text, True, (200, 200, 200))
            self.screen.blit(inst_text, (WIDTH - 120, 10 + i * 20))
        
        pygame.display.flip()
    
    def run(self, fps=60):
        """Main game loop."""
        print("Manual Drive Example")
        print("-" * 60)
        print("Controls:")
        print("  Arrow Keys: Drive (LEFT/RIGHT to steer, UP/DOWN to throttle)")
        print("  SPACE: Toggle between Manual and AI control")
        print("  R: Reset episode")
        print("  Q or ESC: Quit")
        print("-" * 60)
        
        obs = self.reset()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Toggle control mode
                        self.manual_control = not self.manual_control
                        mode = "Manual" if self.manual_control else "AI"
                        print(f"Switched to {mode} control")
                    elif event.key == pygame.K_r:
                        # Reset
                        obs = self.reset()
                        print("Episode reset")
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
            
            # Get action
            if self.manual_control:
                action = self.get_manual_action()
            else:
                action = self.get_action_from_policy(obs)
            
            # Step environment
            next_obs, reward, done = self.step(action)
            obs = next_obs
            
            if done:
                print(f"Episode finished! Reward: {self.episode_reward:.2f}")
                obs = self.reset()
            
            # Render
            self.render()
            self.clock.tick(fps)
        
        pygame.quit()
        print("Thanks for playing!")


def main():
    parser = argparse.ArgumentParser(description='Manual drive example')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frames per second (default: 60)')
    
    args = parser.parse_args()
    
    env = ManualDriveEnv(seed=args.seed)
    env.run(fps=args.fps)


if __name__ == '__main__':
    main()

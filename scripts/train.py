#!/usr/bin/env python3
"""CLI entry point for continuous training."""

import argparse
import os
import sys
import yaml
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_system.training import ContinuousTrainer
from rl_system.config import config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Continuous RL Training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: config/default.yaml)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Maximum number of episodes to run (default: infinite)')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering (disabled in headless mode by default)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no display)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to model file to load')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available saved models')
    
    args = parser.parse_args()
    
    # Determine if headless mode
    headless = args.headless or not args.render
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(current_dir, 'config', 'default.yaml')
    
    print(f"Loading configuration from: {config_path}")
    print(f"Headless mode: {headless}")
    print(f"Max episodes: {args.episodes}")
    
    # Initialize trainer
    trainer = ContinuousTrainer(config_path=config_path, headless=headless)
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        success = trainer.load_model(args.load_model)
        if not success:
            print("Failed to load model, starting fresh")
    
    # List models if requested
    if args.list_models:
        models = trainer.list_models()
        print("Available models:")
        for model in models:
            print(f"  {model}")
        return
    
    # Check configuration
    print("\nConfiguration Summary:")
    print(f"  Environment: {config.get('environment.width', 800)}x{config.get('environment.height', 600)}")
    print(f"  Latent size: {config.get('agent.latent_size', 32)}")
    print(f"  Batch size: {config.get('training.batch_size', 32)}")
    print(f"  Buffer size: {config.get('training.buffer_size', 10000)}")
    print(f"  Initial exploration: {config.get('training.exploration_initial', 1.0)}")
    print(f"  Model save path: {config.get('model.save_path', 'models/')}")
    
    print("\nControls (when rendering):")
    print("  SPACE - Toggle manual/AI control")
    print("  S - Save model")
    print("  ESC - Stop training")
    print("\nStarting training...")
    
    # Start continuous training
    try:
        trainer.run_continuous(max_episodes=args.episodes, render=args.render)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise
    finally:
        print(f"Training completed. Episodes: {trainer.episode_counter}, Steps: {trainer.step_counter}")
        
        # Save final model
        final_path = os.path.join(trainer.model_save_path, f"{trainer.filename_prefix}final.pkl")
        trainer.save_model("final")
        print(f"Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
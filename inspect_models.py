#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime

# Ensure we can import rl_system
sys.path.append(os.getcwd())

from rl_system.persistence import PersistenceManager

def format_timestamp(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description="Inspect available model versions")
    parser.add_argument('--models-dir', default='models/', help='Directory containing models')
    parser.add_argument('--sort', choices=['time', 'reward', 'progress'], default='time', help='Sort order')
    
    args = parser.parse_args()
    
    pm = PersistenceManager(args.models_dir)
    models = pm.list_models()
    
    if not models:
        print("No models found.")
        return

    # Sort based on argument
    if args.sort == 'time':
        models.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
    elif args.sort == 'reward':
        models.sort(key=lambda x: x.get('mean_reward', 0), reverse=True)
    elif args.sort == 'progress':
        models.sort(key=lambda x: x.get('best_progress', 0), reverse=True)

    print(f"{'Filename':<35} | {'Created At':<25} | {'Ep':<5} | {'Reward':<10} | {'Progress':<10}")
    print("-" * 95)
    
    for m in models:
        filename = m.get('filename', 'unknown')
        created_at = m.get('created_at', format_timestamp(m.get('timestamp', 0)))
        episode = m.get('episode_count', -1)
        reward = m.get('mean_reward', 0.0)
        progress = m.get('best_progress', 0.0)
        
        print(f"{filename:<35} | {created_at:<25} | {episode:<5} | {reward:<10.2f} | {progress:<10.2f}")

if __name__ == "__main__":
    main()

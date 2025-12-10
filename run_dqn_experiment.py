#!/usr/bin/env python3
"""Run DQN experiment on car racing environment."""

import sys
sys.path.insert(0, '/workspaces/RL')

from experiments import RLExperiment

def main():
    """Run DQN experiment."""
    print("="*60)
    print("Deep Q-Network (DQN) Experiment")
    print("="*60)
    
    # Environment config
    env_config = {
        'max_steps': 500
    }
    
    # Algorithm config
    algorithm_config = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'buffer_size': 5000,
        'batch_size': 32,
        'target_update_freq': 100
    }
    
    # Create experiment
    exp = RLExperiment(
        algorithm_name='dqn',
        env_config=env_config,
        algorithm_config=algorithm_config,
        experiment_name='dqn_car_racing',
        seed=42
    )
    
    # Run experiment
    results = exp.run(
        num_episodes=50,
        max_steps_per_episode=500,
        eval_interval=5,
        eval_episodes=5,
        save_interval=10
    )
    
    print("\n" + "="*60)
    print("Experiment Results")
    print("="*60)
    print(f"Algorithm: {results['algorithm']}")
    print(f"Final Mean Reward: {results['final_eval']['mean_reward']:.2f}")
    print(f"Final Std Reward: {results['final_eval']['std_reward']:.2f}")
    print(f"Log Directory: {results['log_dir']}")


if __name__ == '__main__':
    main()

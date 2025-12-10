#!/usr/bin/env python3
"""Run Policy Gradient experiment on car racing environment."""

import sys
sys.path.insert(0, '/workspaces/RL')

from experiments import RLExperiment

def main():
    """Run Policy Gradient experiment."""
    print("="*60)
    print("Policy Gradient (REINFORCE) Experiment")
    print("="*60)
    
    # Environment config
    env_config = {
        'max_steps': 500
    }
    
    # Algorithm config
    algorithm_config = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'batch_size': 32
    }
    
    # Create experiment
    exp = RLExperiment(
        algorithm_name='policy_gradient',
        env_config=env_config,
        algorithm_config=algorithm_config,
        experiment_name='pg_car_racing',
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

# RL Algorithms & Experimentation Suite

A comprehensive collection of RL algorithms with modular environments, evaluation tools, and reproducible experimentation pipelines.

## Features

✅ **Core RL Methods Implemented:**
- Q-Learning (Tabular)
- Deep Q-Network (DQN)
- Policy Gradients (REINFORCE)
- World Models + Actor-Critic

✅ **Modular Design:**
- BaseRLEnvironment abstract class for custom environments
- CarRacingEnvironment reference implementation
- Algorithm-agnostic training pipeline

✅ **Reproducibility:**
- Seed management for deterministic experiments
- Config-based experiment tracking
- Automatic checkpoint saving
- Metrics logging to disk

✅ **Evaluation Tools:**
- Metrics tracker (rewards, losses, exploration rates)
- Experiment logger with config tracking
- Evaluator for performance assessment
- Multi-episode evaluation support

## Project Structure

```
RL/
├── algorithms/           # RL algorithms
│   ├── qlearning.py     # Q-Learning
│   ├── dqn.py          # Deep Q-Network
│   ├── policy_gradient.py # Policy Gradient (REINFORCE)
│   └── world_model.py   # World Model Agent
├── environments/         # RL environments
│   ├── base_env.py      # Abstract base environment
│   └── car_env.py       # Car racing environment
├── experiments/          # Experimentation pipeline
│   └── experiment.py    # RLExperiment class
├── evaluation/           # Evaluation tools
│   └── metrics.py       # Metrics, Logger, Evaluator
├── run_dqn_experiment.py  # DQN experiment script
├── run_pg_experiment.py   # Policy Gradient experiment script
└── run_wm_experiment.py   # World Model experiment script
```

## Usage

### Running Experiments

```python
from experiments import RLExperiment

# Create experiment
exp = RLExperiment(
    algorithm_name='dqn',
    env_config={'max_steps': 500},
    algorithm_config={'learning_rate': 1e-3, 'gamma': 0.99},
    experiment_name='dqn_test',
    seed=42
)

# Run training
results = exp.run(
    num_episodes=100,
    max_steps_per_episode=500,
    eval_interval=10,
    eval_episodes=5,
    save_interval=50
)
```

### Running Pre-built Experiments

```bash
# Run DQN experiment
python run_dqn_experiment.py

# Run Policy Gradient experiment
python run_pg_experiment.py

# Run World Model experiment
python run_wm_experiment.py
```

## Algorithm Details

### Q-Learning
- Tabular method for discrete state/action spaces
- Epsilon-greedy exploration
- Experience-based incremental updates

### Deep Q-Network (DQN)
- Neural network approximator for Q-values
- Experience replay buffer
- Target network for stability
- Adam optimizer

### Policy Gradient (REINFORCE)
- Direct policy optimization
- Entropy regularization for exploration
- Return normalization for stability
- Log-likelihood based loss

### World Model + Actor
- Learns world dynamics model
- Actor network learns policy conditioned on latent states
- Multi-component loss (reconstruction + reward prediction)
- JAX-based gradient computation

## Evaluation & Metrics

The evaluation system provides:

- **Metrics Tracking**: Episode rewards, lengths, losses, exploration rates
- **Experiment Logging**: Config, checkpoints, metrics saved to disk
- **Performance Evaluation**: Multi-episode mean, std, max/min rewards
- **Interpretability**: Tracked Q-values, policy distributions (where applicable)

## Reproducibility

All experiments ensure reproducibility through:
1. Seed management (environment and algorithms)
2. Deterministic algorithm implementations
3. Config-based parameter tracking
4. Automatic metrics logging
5. Checkpoint saving

## Example Workflow

```python
from algorithms import DeepQNetwork
from environments import CarRacingEnvironment
from evaluation import Metrics, Evaluator

# Initialize components
env = CarRacingEnvironment(seed=42, max_steps=500)
agent = DeepQNetwork(
    state_size=env.observation_shape[0],
    action_size=env.action_shape[0]
)
metrics = Metrics()
evaluator = Evaluator(env, metrics)

# Training loop
for episode in range(100):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs, training=True)
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        loss = agent.train_step()
        metrics.record_loss(loss)
        obs = next_obs
    
    metrics.record_episode(env.episode_reward, env.current_step)

# Evaluate
eval_results = evaluator.evaluate(agent, num_episodes=20)
print(f"Mean Reward: {eval_results['mean_reward']:.2f}")
```

## Next Steps

- [ ] Add A3C for distributed training
- [ ] Add PPO for continuous action spaces
- [ ] Add DDPG for continuous control
- [ ] Add intrinsic motivation (curiosity-driven)
- [ ] Add transfer learning capabilities
- [ ] Add visualization dashboards

## License

MIT

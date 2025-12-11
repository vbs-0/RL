import logging
import json
import csv
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List, Optional


class MetricsCollector:
    """Structured monitoring layer for RL training metrics."""
    
    def __init__(self, log_dir: str = 'logs'):
        """Initialize metrics collector with rotating file handlers.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('rl_training')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler with rotation (10 MB max, keep 5 backups)
        log_file = self.log_dir / 'training.log'
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # JSONL file for structured metrics
        self.jsonl_file = self.log_dir / 'metrics.jsonl'
        
        # CSV file for time-series analysis
        self.csv_file = self.log_dir / 'metrics.csv'
        self.csv_headers = None
        self.csv_writer = None
        
        # In-memory metrics for recent history
        self.episode_metrics: List[Dict[str, Any]] = []
        self.max_history = 1000  # Keep last 1000 episodes
        
        # Startup time for uptime calculation
        self.startup_time = datetime.now()
        
    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """Log metrics for a completed episode.
        
        Args:
            episode_data: Dictionary containing episode metrics
                Expected keys: episode, reward, actor_loss, world_model_loss,
                steps, exploration_rate, and optionally other metrics
        """
        # Add timestamp
        episode_data['timestamp'] = datetime.now().isoformat()
        episode_data['uptime_seconds'] = (
            datetime.now() - self.startup_time
        ).total_seconds()
        
        # Store in memory
        self.episode_metrics.append(episode_data.copy())
        if len(self.episode_metrics) > self.max_history:
            self.episode_metrics.pop(0)
        
        # Log to text file
        self.logger.info(
            f"Episode {episode_data.get('episode', 'N/A')}: "
            f"Reward={episode_data.get('reward', 0.0):.2f}, "
            f"Actor Loss={episode_data.get('actor_loss', 0.0):.4f}, "
            f"World Model Loss={episode_data.get('world_model_loss', 0.0):.4f}, "
            f"Steps={episode_data.get('steps', 0)}, "
            f"Exploration Rate={episode_data.get('exploration_rate', 0.0):.4f}"
        )
        
        # Write to JSONL file
        self._write_jsonl(episode_data)
        
        # Write to CSV file
        self._write_csv(episode_data)
    
    def _write_jsonl(self, data: Dict[str, Any]) -> None:
        """Write metric data to JSONL file.
        
        Args:
            data: Dictionary of metrics
        """
        try:
            with open(self.jsonl_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')
        except IOError as e:
            self.logger.error(f"Failed to write JSONL: {e}")
    
    def _write_csv(self, data: Dict[str, Any]) -> None:
        """Write metric data to CSV file.
        
        Args:
            data: Dictionary of metrics
        """
        try:
            # Determine headers from first write
            if self.csv_headers is None:
                self.csv_headers = sorted(data.keys())
            
            file_exists = self.csv_file.exists()
            
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write row with available data
                row = {k: data.get(k, '') for k in self.csv_headers}
                writer.writerow(row)
        except IOError as e:
            self.logger.error(f"Failed to write CSV: {e}")
    
    def log_training_step(self, step_data: Dict[str, Any]) -> None:
        """Log a training step (more frequent than episode logging).
        
        Args:
            step_data: Dictionary containing step-level metrics
        """
        step_data['timestamp'] = datetime.now().isoformat()
        self.logger.debug(f"Training step: {step_data}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get recent metrics history.
        
        Returns:
            List of recent episode metrics
        """
        return self.episode_metrics.copy()
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent episode metrics.
        
        Returns:
            Latest episode data or None if no episodes logged
        """
        if self.episode_metrics:
            return self.episode_metrics[-1].copy()
        return None
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Calculate aggregated statistics from recent history.
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not self.episode_metrics:
            return {}
        
        rewards = [m.get('reward', 0.0) for m in self.episode_metrics]
        actor_losses = [m.get('actor_loss', 0.0) for m in self.episode_metrics]
        world_losses = [m.get('world_model_loss', 0.0) 
                       for m in self.episode_metrics]
        exploration_rates = [m.get('exploration_rate', 0.0) 
                            for m in self.episode_metrics]
        steps = [m.get('steps', 0) for m in self.episode_metrics]
        
        total_episodes = len(self.episode_metrics)
        
        return {
            'total_episodes': total_episodes,
            'avg_reward': sum(rewards) / total_episodes if rewards else 0.0,
            'max_reward': max(rewards) if rewards else 0.0,
            'min_reward': min(rewards) if rewards else 0.0,
            'avg_actor_loss': sum(actor_losses) / total_episodes 
                if actor_losses else 0.0,
            'avg_world_loss': sum(world_losses) / total_episodes 
                if world_losses else 0.0,
            'avg_exploration_rate': sum(exploration_rates) / total_episodes 
                if exploration_rates else 0.0,
            'total_steps': sum(steps),
            'uptime_seconds': (
                datetime.now() - self.startup_time
            ).total_seconds(),
        }
    
    def load_metrics_from_disk(self) -> None:
        """Load previously logged metrics from JSONL file."""
        if not self.jsonl_file.exists():
            return
        
        try:
            with open(self.jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.episode_metrics.append(data)
            
            # Keep only recent history
            if len(self.episode_metrics) > self.max_history:
                self.episode_metrics = self.episode_metrics[-self.max_history:]
            
            self.logger.info(
                f"Loaded {len(self.episode_metrics)} metrics from disk"
            )
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load metrics from disk: {e}")

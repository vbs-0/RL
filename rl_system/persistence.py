import os
import json
import pickle
import time
from typing import Dict, Any, List, Optional

class PersistenceManager:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.registry_path = os.path.join(model_dir, "registry.json")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self._ensure_registry()

    def _ensure_registry(self):
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump({}, f)

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
             return {}
        with open(self.registry_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _save_registry(self, registry: Dict[str, Any]):
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=4)

    def save_model(self, params: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        # Generate filename
        timestamp = int(time.time())
        episode = metadata.get('episode_count', 0)
        filename = f"model_ep{episode}_{timestamp}.pkl"
        filepath = os.path.join(self.model_dir, filename)

        # Save weights
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

        # Update registry
        registry = self._load_registry()
        registry_entry = {
            'filename': filename,
            'timestamp': timestamp,
            'episode_count': episode,
            'mean_reward': metadata.get('mean_reward', 0.0),
            'best_progress': metadata.get('best_progress', 0.0),
            'created_at': time.ctime(timestamp)
        }
        registry[filename] = registry_entry
        self._save_registry(registry)
        
        return filename

    def load_model(self, identifier: str) -> Optional[Any]:
        # identifier can be filename or 'best'
        registry = self._load_registry()
        
        target_filename = None
        if identifier == 'best':
            if not registry:
                # If registry is empty, try to find pickle files manually? 
                # Or just return None.
                # Let's fall back to listing files if registry is empty but files exist?
                # But for 'best', we need metrics. If no registry, we don't know which is best.
                # Maybe fallback to parsing filenames if they follow a pattern, but that's risky.
                # Let's assume registry is the source of truth for 'best'.
                return None
            
            # Find model with highest best_progress
            target_filename = max(registry.values(), key=lambda x: x.get('best_progress', -float('inf')))['filename']
        elif identifier in registry:
             target_filename = identifier
        else:
             # Check if file exists even if not in registry (legacy support or direct filename)
             filepath = os.path.join(self.model_dir, identifier)
             if os.path.exists(filepath):
                 target_filename = identifier

        if target_filename:
             filepath = os.path.join(self.model_dir, target_filename)
             if os.path.exists(filepath):
                 with open(filepath, 'rb') as f:
                     return pickle.load(f)
        
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        registry = self._load_registry()
        # Ensure we also list files that might not be in registry (legacy)
        files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        models = []
        for f in files:
            if f in registry:
                models.append(registry[f])
            else:
                # Basic info for legacy models
                filepath = os.path.join(self.model_dir, f)
                timestamp = os.path.getmtime(filepath)
                models.append({
                    'filename': f,
                    'timestamp': timestamp,
                    'episode_count': -1, # Unknown
                    'mean_reward': 0.0,
                    'best_progress': 0.0,
                    'created_at': time.ctime(timestamp)
                })
        
        # Sort by timestamp desc
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models

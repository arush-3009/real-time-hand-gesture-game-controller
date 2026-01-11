import yaml
import os


class Config:

    def __init__(self, config_file="config.yml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):

        if not os.path.exists(self.config_file):
            print(f"Warning: {self.config_file} not found, using defaults")
            return self._get_defaults()
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                print(f"Loaded configuration from {self.config_file}")
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return self._get_defaults()
        
    
    def _get_defaults(self):
        
        default_config = {
            "camera": {
                "width": 640,
                "height": 480,
                "device_id": 0
            },
            "gestures": {
                "left_threshold": 0.40,
                "right_threshold": 0.60,
                "smoothing_threshold": 3,
                "brake_to_reverse_delay": 1.5
            },
            "display": {
                "mode": "info_only",
                "show_fps": True,
                "show_gesture_table": True,
                "show_steering_bars": True,
                "show_keys": True,
                "window_name": "Asphalt Hand Control"
            },
            "debug": {
                "enabled": False,
                "log_to_file": False,
                "log_file": "debug.log"
            }
        }

        return default_config
    
    def get(self, *keys):
        """
        Safely get nested configuration value
        
        Args:
            *keys: Sequence of keys to traverse (e.g., 'camera', 'width')
        
        Returns:
            Configuration value or None if not found
            
        Example:
            config.get('camera', 'width')  # Returns 640
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None
        return value
    
    def save(self):
        """
        Save current configuration back to YAML file
        """
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")

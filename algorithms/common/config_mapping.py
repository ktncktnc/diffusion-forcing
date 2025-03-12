import os
import json
import random
import importlib

class ConfigMapping:
    """
    Class to map objects based on their class and configuration to a value
    Used to map algorithm classes with specific configurations to their respective names to store hidden states for training correction diffusion
    """
    def __init__(self, cfg):
        # List to store mappings
        self.cfg = cfg
        self.root_dir = cfg.root_dir
        self.metatdata_path = os.path.join(self.root_dir, 'metadata.json')
        self.mappings = []
    
    def register(self, cls, config_dict, mapped_path=None):
        """
        Register a mapping between a class with a specific config dictionary to a mapped value
        
        Args:
            cls: The class to map
            config_dict: Dictionary of configuration key-value pairs to match
            mapped_value: The value to map to when this combination is found
        """
        self.mappings.append((cls, config_dict, mapped_path))
    
    def get_value(self, obj):
        """
        Get the mapped value for an object instance based on its class and configuration
        
        Args:
            obj: The instance to look up
        
        Returns:
            The mapped value. If no mapping exists, a new path is generated and returned
        """
        cls = obj.__class__
        configs = cls.mapping_config()

        for (mapped_cls, config_dict, mapped_value) in self.mappings:
            # Check if the object is an instance of the mapped class
            if not isinstance(obj, mapped_cls):
                continue
                
            # Check if all config key-value pairs match
            match = True
            for config_key, config_value in config_dict.items():
                if not hasattr(obj, config_key) or getattr(configs, config_key) != config_value:
                    match = False
                    break
            
            if match:
                return mapped_value
        
        new_mapped_path = os.path.join(self.root_dir, f"{cls.__name__}_{random.randint(0, 1000000)}")
        self.register(cls, configs, new_mapped_path)
        self.save_to_json()
        return new_mapped_path
    
    def save_to_json(self):
        """
        Save the mapping configuration to a JSON file
        """
        serializable_mappings = []
        
        for (cls, config_dict, mapped_value) in self.mappings:
            # Save the class as its module path and name for later import
            cls_info = {
                "module": cls.__module__,
                "name": cls.__name__
            }
            
            serializable_mappings.append({
                "class": cls_info,
                "config": config_dict,
                "value": mapped_value
            })
        
        with open(self.metatdata_path, 'w') as f:
            json.dump(serializable_mappings, f, indent=2)
    
    @classmethod
    def load_from_json(cls, cfg):
        """
        Load mapping configuration from a JSON file
        
        Returns:
            A new ConfigMapping instance with the loaded mappings
        """
        mapping = cls(cfg)
        
        try:
            with open(mapping.metatdata_path, 'r') as f:
                serialized_mappings = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Couldn't find metadata file at {mapping.metatdata_path}. Creating new mapping.")
            mapping.save_to_json()
            return mapping
        
        for item in serialized_mappings:
            # Import the class dynamically
            module_path = item["class"]["module"]
            class_name = item["class"]["name"]
            
            try:
                module = importlib.import_module(module_path)
                class_obj = getattr(module, class_name)
                
                # Register the mapping
                mapping.register(
                    class_obj, 
                    item["config"], 
                    item["value"]
                )
            except (ImportError, AttributeError) as e:
                print(f"Warning: Couldn't load class {class_name} from {module_path}: {e}")
        
        return mapping
    
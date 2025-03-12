import json
import importlib

class ConfigMapping:
    def __init__(self):
        # List to store mappings
        self.mappings = []
    
    def register(self, cls, config_dict, mapped_value):
        """
        Register a mapping between a class with a specific config dictionary to a mapped value
        
        Args:
            cls: The class to map
            config_dict: Dictionary of configuration key-value pairs to match
            mapped_value: The value to map to when this combination is found
        """
        self.mappings.append((cls, config_dict, mapped_value))
    
    def get_value(self, obj):
        """
        Get the mapped value for an object instance based on its class and configuration
        
        Args:
            obj: The instance to look up
        
        Returns:
            The mapped value or None if no mapping exists
        """
        cls = obj.__class__
        
        for (mapped_cls, config_dict, mapped_value) in self.mappings:
            # Check if the object is an instance of the mapped class
            if not isinstance(obj, mapped_cls):
                continue
                
            # Check if all config key-value pairs match
            match = True
            for config_key, config_value in config_dict.items():
                if not hasattr(obj, config_key) or getattr(obj, config_key) != config_value:
                    match = False
                    break
            
            if match:
                return mapped_value
        
        return None
    
    def save_to_json(self, filepath):
        """
        Save the mapping configuration to a JSON file
        
        Args:
            filepath: Path to save the JSON file
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
        
        with open(filepath, 'w') as f:
            json.dump(serializable_mappings, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath):
        """
        Load mapping configuration from a JSON file
        
        Args:
            filepath: Path to the JSON file
        
        Returns:
            A new ConfigMapping instance with the loaded mappings
        """
        mapping = cls()
        
        with open(filepath, 'r') as f:
            serialized_mappings = json.load(f)
        
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
    


# Example usage
class DataProcessor:
    def __init__(self, mode="standard", level=1, cache=False, threads=4):
        self.mode = mode
        self.level = level
        self.cache = cache
        self.threads = threads

# Create mapping registry
# mapping = ConfigMapping()
mapping = ConfigMapping.load_from_json("processor_mappings.json")
# mapping.save_to_json("processor_mappings.json")


# Register mappings with multiple config keys
# mapping.register(
#     DataProcessor, 
#     {"mode": "fast", "threads": 8}, 
#     "High-performance parallel processing"
# )

# mapping.register(
#     DataProcessor, 
#     {"mode": "accurate", "level": 3, "cache": True}, 
#     "High-precision processing with caching"
# )

# mapping.register(
#     DataProcessor, 
#     {"level": 2, "cache": False}, 
#     "Standard processing without caching"
# )

# # Save the mapping to a JSON file
# mapping.save_to_json("processor_mappings.json")

# Test with objects
processor1 = DataProcessor(mode="fast", threads=8)
processor2 = DataProcessor(mode="accurate", level=3, cache=True)
processor3 = DataProcessor(level=2, cache=False)
processor4 = DataProcessor()  # Default config

print(mapping.get_value(processor1))  # "High-performance parallel processing"
print(mapping.get_value(processor2))  # "High-precision processing with caching"
print(mapping.get_value(processor3))  # "Standard processing without caching"
print(mapping.get_value(processor4))  # None (no matching config)
import os
import yaml

def load_config(config):
    config = os.path.join("config", config)
    with open(config, 'r') as config_file:
        return yaml.load(config_file)



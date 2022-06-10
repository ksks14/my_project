import pathlib
import yaml

# get the root
BASE_DIR = pathlib.Path(__file__).parent.parent
config_path = BASE_DIR / 'config' / 'system.yaml'


# get the config
def get_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


# successfully get the config
config = get_config(config_path)

from omegaconf import OmegaConf

def load_deploy_config(config_path: str):
    config = OmegaConf.load(config_path)
    return config
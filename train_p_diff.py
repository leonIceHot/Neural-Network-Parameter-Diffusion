import hydra
from omegaconf import DictConfig
from core.runner.runner import *
import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    if config.mode == 'train':
        result = train_generation(config)
    elif config.mode == 'test':
        result = test_generation(config)

if __name__ == "__main__":
    training_for_data()
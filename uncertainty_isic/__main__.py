import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import clearml

import torch
import torch.nn as nn 

import albumentations as A

from pipeline.train import ISICTrainingPipeline
from pipeline.transforms.augmentation import get_transforms

# Modules
from modules.data import ISICDataModule
from modules.orchestration import ClassModel

import gc

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    datamodule = ISICDataModule(csv_file = cfg.csv_file,
                                img_dir = cfg.img_dir,
                                transform = get_transforms(cfg.img_size),
                                batch_size = cfg.batch_size,
                                num_workers = cfg.num_workers)
    
    model = instantiate(cfg.model)
    criterion = instantiate(cfg.criterion)
    
    module = ClassModel(model = model,
                        criterion = criterion)
    
    training_pipeline = ISICTrainingPipeline(datamodule=datamodule,
                                             module=module,
                                             max_epochs=cfg.max_epochs,
                                             cfg=cfg)
    
    training_pipeline.train()
    

if __name__ == '__main__':
    main()

import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import clearml
from clearml import Task, Logger, StorageManager

import torch
import torch.nn as nn 

import albumentations as A

import pandas as pd
import numpy as np

import glob

import torchvision

import yaml

# Pipeline
from pipeline.train import ISICTrainingPipeline
from pipeline.inference import ISICInferencePipeline
from pipeline.transforms.augmentation import get_transforms

# Modules
from modules.data import CIFAR10DataModule
from modules.orchestration import ClassModel

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

import random 
import os

import pytorch_lightning as pl


import gc

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kfold = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    transforms = get_transforms(cfg.img_size)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms['train'])
    
    valset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms['valid'])
    
    labels = [trainset[i][1] for i in range(len(trainset))]
    
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(trainset)), labels)):

        print(f'Fold: {fold}')

        train_subset = torch.utils.data.Subset(trainset, train_idx)
        val_subset = torch.utils.data.Subset(valset, val_idx)

        datamodule = CIFAR10DataModule(train_set = train_subset,
                                    val_set = val_subset,
                                    batch_size = cfg.batch_size,
                                    num_workers = cfg.num_workers,
                                    seed = cfg.seed)
        
        model = instantiate(cfg.model)
        criterion = instantiate(cfg.criterion)

        module = ClassModel(model = model,
                            criterion = criterion,
                            lr = cfg.lr,
                            weight_decay = cfg.weight_decay,
                            num_epochs = cfg.max_epochs)
        
        
        if cfg.checkpoint_path:
            best_ckpt_path = cfg.checkpoint_path
            training_pipeline = None
            checkpoint = torch.load(best_ckpt_path, weights_only=True)
            module.load_state_dict(checkpoint['state_dict'])
            module = module.to(device)
            datamodule.setup()
        else: 
            module = module.to(device)
            training_pipeline = ISICTrainingPipeline(datamodule=datamodule,
                                                    module=module,
                                                    max_epochs=cfg.max_epochs,
                                                    cfg=cfg,
                                                    fold=fold,
                                                    patience=cfg.patience)
            
            best_ckpt_path = training_pipeline.train()
            checkpoint = torch.load(best_ckpt_path, weights_only=True)
            module.load_state_dict(checkpoint['state_dict'])
            torch.save(module, f'{cfg.name}_{fold}.pth')
            module = module.to(device)

            del model
            del datamodule
            del criterion
            del module
            del training_pipeline

            gc.collect()


if __name__ == '__main__':
    main()
    gc.collect()
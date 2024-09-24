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

import yaml

# Pipeline
from pipeline.train import ISICTrainingPipeline
from pipeline.inference import ISICInferencePipeline
from pipeline.transforms.augmentation import get_transforms

# Modules
from modules.data import ISICDataModule
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

def get_file_with_fold(directory, fold):
    # Create the pattern for the file name
    pattern = os.path.join(directory, f'*{fold}.pth')
    
    # Use glob to find files matching the pattern
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"No file found matching the pattern '*{fold}.pth' in directory '{directory}'")
        return None
    
    if len(matching_files) > 1:
        print(f"Multiple files found matching the pattern '*{fold}.pth'. Returning the first one.")
    
    # Return the first matching file
    return matching_files[0]

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kfold = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    full_dataset = pd.read_csv('/repo/tta_class/data/dataset.csv')
    full_dataset['target'] = full_dataset['target'].astype(int)

    if cfg.debug: 
        # Limiting dataset for testing purposes
        df_positive = full_dataset[full_dataset['target'] == 1].iloc[:100, :]
        df_negative = full_dataset[full_dataset['target'] == 0].iloc[:100, :]
        
        full_dataset = pd.concat([df_positive, df_negative], ignore_index=True)
    
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset, full_dataset['target'])):

        print(f'Fold: {fold}')

        train_df = full_dataset.iloc[train_idx, :]
        print(f'Train: {train_df['target'].value_counts()}')
        val_df = full_dataset.iloc[val_idx, :]
        print(f'Valid: {val_df['target'].value_counts()}')

        datamodule = ISICDataModule(train_df = train_df,
                                    val_df = val_df,
                                    img_dir = cfg.img_dir,
                                    transform = get_transforms(cfg.img_size),
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
        
        #checkpoint = torch.load(f'{cfg.dir_path}')
        
        # Example usage
        directory = cfg.dir_path

        file_path = get_file_with_fold(directory, fold)
        if file_path:
            print(f"Found file: {file_path}")

        torch.serialization.add_safe_globals([ClassModel])
        module = torch.load(file_path)

        print('Checkpoint loaded')


if __name__ == '__main__':
    main()
    gc.collect()
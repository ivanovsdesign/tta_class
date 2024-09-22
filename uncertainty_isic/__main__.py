import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import clearml
from clearml import Task, Logger

import torch
import torch.nn as nn 

import albumentations as A

import pandas as pd
import numpy as np

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


import gc

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    task = Task.init(project_name='isic_balanced',
                task_name='debug' if cfg.debug else f'{cfg.name}')
    kfold = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    full_dataset = pd.read_csv('/repo/tta_class/data/dataset.csv')
    full_dataset['target'] = full_dataset['target'].astype(int)

    if cfg.debug: 
        # Limiting dataset for testing purposes
        df_positive = full_dataset[full_dataset['target'] == 1].iloc[:100, :]
        df_negative = full_dataset[full_dataset['target'] == 0].iloc[:100, :]
        
        full_dataset = pd.concat([df_positive, df_negative], ignore_index=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset, full_dataset['target'])):

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
                            criterion = criterion)
        
        if cfg.checkpoint_path:
            best_ckpt_path = torch.load(cfg.checkpoint_path)
            checkpoint = torch.load(best_ckpt_path, weights_only=True)
            module.load_state_dict(checkpoint['state_dict'])
        else: 
            training_pipeline = ISICTrainingPipeline(datamodule=datamodule,
                                                    module=module,
                                                    max_epochs=cfg.max_epochs,
                                                    cfg=cfg)
            
            best_ckpt_path = training_pipeline.train()
            checkpoint = torch.load(best_ckpt_path, weights_only=True)
            module.load_state_dict(checkpoint['state_dict'])
            #torch.save(module, f'{cfg.name}_{fold}.pth')


        inference_pipeline = ISICInferencePipeline(datamodule=datamodule,
                                                    module=module)
        preds_no_tta, targets_no_tta = inference_pipeline.inference_without_tta()
        metrics_no_tta = inference_pipeline.compute_metrics(preds = preds_no_tta,
                                                            targets = targets_no_tta,
                                                            prefix = 'no_tta')
    
        np.save(f'preds_{cfg.name}_{fold}.npy', preds_no_tta)
        np.save(f'targets_{cfg.name}_{fold}.npy', targets_no_tta)

        metrics = {
            'name' : cfg.name,
            'fold' : fold,
            'checkpoint_path' : best_ckpt_path,
            **metrics_no_tta
        }

        metrics = pd.DataFrame([metrics])

        numeric_columns = metrics.columns[3:]
        metrics[numeric_columns] = metrics[numeric_columns].astype(float)

        Logger.current_logger().report_table(
            title=f'{cfg.name}_fold{fold}', 
            series="PD with index", 
            iteration=fold, 
            table_plot=metrics
        )

        del model
        del datamodule
        del criterion
        del module
        del training_pipeline
        del inference_pipeline

        gc.collect()


if __name__ == '__main__':
    main()

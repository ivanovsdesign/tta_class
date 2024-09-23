import torch
from torch.utils.data import DataLoader
from typing import Callable
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from omegaconf import DictConfig

class ISICTrainingPipeline():
    def __init__(self,
                 datamodule: Callable,
                 module: Callable,
                 **kwargs) -> None:
        
        self.__dict__.update(kwargs)
        self.datamodule = datamodule
        self.module = module

    def setup():
        pass
    def train(self, **kwargs):
        checkpoint_callback = ModelCheckpoint(dirpath='checkpoints',
                                        monitor='val_loss',
                                        save_top_k=1,
                                        mode='min',
                                        filename=f'{self.cfg.name}_{self.cfg.model.backbone}'+'{epoch:02d}-{val_acc:.2f}'+f'{self.cfg.seed}_{self.fold}',
                                        verbose=True
                                        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,  # Number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'
        )
        trainer = Trainer(
            max_epochs=self.max_epochs,  # Number of epochs to train for
            logger=True,  # Use default logger (TensorBoard)
            callbacks=[LearningRateMonitor(logging_interval='epoch'),
                       checkpoint_callback,
                       early_stopping_callback
                        ])

        # Train the module
        trainer.fit(self.module, datamodule=self.datamodule)

        return checkpoint_callback.best_model_path
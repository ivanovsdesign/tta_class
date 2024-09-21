import torch
from torch.utils.data import DataLoader
from typing import Callable
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor

class ISICTrainingPipeline():
    def __init__(self,
                 datamodule: Callable,
                 module: Callable,
                 transforms: Callable,
                 **kwargs) -> None:
        
        self.__dict__.update(kwargs)
        self.datamodule = datamodule
        self.module = module
        self.transforms = transforms

    def setup():
        pass
    def train(self, **kwargs):
        trainer = Trainer(
            max_epochs=self.max_epochs,  # Number of epochs to train for
            progress_bar_refresh_rate=20,  # Refresh the progress bar every 20 batches
            logger=True,  # Use default logger (TensorBoard)
            checkpoint_callback=True,  # Save checkpoints
            callbacks=[LearningRateMonitor(logging_interval='epoch')]
        )

        # Train the module
        trainer.fit(self.module, datamodule=self.datamodule)
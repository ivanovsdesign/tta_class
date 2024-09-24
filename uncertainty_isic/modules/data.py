import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd

from datasets.isic_balanced import ISICDataset

class ISICDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 img_dir: str,
                 transform: bool = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seed: int = 42):
        '''
        Data Module for ISISC Balanced dataset
        `is_tta` flag ensures transforms for test time augmentations are being used
        '''
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.img_dir = img_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        self.train_dataset = ISICDataset(self.train_df, self.img_dir, transform=self.transform['train'])
        self.val_dataset = ISICDataset(self.val_df, self.img_dir, transform=self.transform['valid'])
        self.test_dataset = ISICDataset(self.val_df, self.img_dir, transform=self.transform['valid'])
        self.tta_dataset = ISICDataset(self.val_df, self.img_dir, transform=self.transform['tta'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def tta_dataloader(self):
        return DataLoader(self.tta_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_set: Subset,
                 val_set: Subset,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 seed: int = 42):
        '''
        Data Module for ISISC Balanced dataset
        `is_tta` flag ensures transforms for test time augmentations are being used
        '''
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def tta_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import pandas as pd

from datasets.isic_balanced import ISICDataset

class ISICDataModule(pl.LightningDataModule):
    def __init__(self,
                 csv_file: str,
                 img_dir: str,
                 transform: bool = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 is_tta: bool = False):
        '''
        Data Module for ISISC Balanced dataset
        `is_tta` flag ensures transforms for test time augmentations are being used
        '''
        super().__init__()
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_tta = is_tta

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        df['target'] = df['target'].astype(int)
        train_df, val_test_df = train_test_split(df, test_size=0.4, stratify=df['target'], random_state=42)
        val_df, test_df = train_test_split(val_test_df, test_size=0.5, stratify=val_test_df['target'], random_state=42)
        
        self.train_dataset = ISICDataset(train_df, self.img_dir, transform=self.transform['train'])
        self.val_dataset = ISICDataset(val_df, self.img_dir, transform=self.transform['valid'])
        if self.is_tta == False:
            self.test_dataset = ISICDataset(test_df, self.img_dir, transform=self.transform['valid'])
        else:
            self.test_dataset = ISICDataset(test_df, self.img_dir, transform=self.transform['tta'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
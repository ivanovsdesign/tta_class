import pandas as pd
import torch
from typing import List
import cv2
import os

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class ISICDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 img_dir: str,
                 transform: List = None):
        """
        Dataset class for loading ISIC Balanced Dataset
        Can be downloaded via kaggle api: olegopoly/isic-balanced
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing the isic_id and target columns.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        isic_id = self.dataframe.iloc[idx]['isic_id']
        target = self.dataframe.iloc[idx]['target']        
        img_name = os.path.join(self.img_dir, f"{isic_id}.jpg")
        
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Convert image to PyTorch tensor
        image = ToTensor()(image)
        
        # Convert target to long tensor
        target = torch.tensor(target, dtype=torch.long)
        
        return image, target
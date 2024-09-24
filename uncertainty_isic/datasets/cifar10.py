import pandas as pd
import torch
from typing import List
import os
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class CIFAR10Dataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 img_dir: str,
                 transform: List = None):
        """
        Dataset class for loading CIFAR-10 Dataset
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing the filename and label columns.
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

        filename = self.dataframe.iloc[idx]['filename']
        label = self.dataframe.iloc[idx]['label']        
        img_path = os.path.join(self.img_dir, filename)
        
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Convert image to PyTorch tensor
        image = ToTensor()(image)
        
        # Convert label to long tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
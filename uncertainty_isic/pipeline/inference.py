import torch
from torch.utils.data import DataLoader
from typing import Callable
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve
import numpy as np
from sklearn.metrics import auc

import numpy as np

from omegaconf import DictConfig

class ISICInferencePipeline():
    def __init__(self,
                 datamodule: Callable,
                 module: Callable,
                 **kwargs) -> None:
        
        self.__dict__.update(kwargs)
        self.datamodule = datamodule
        self.module = module

    def setup():
        pass

    def inference_without_tta(self):
        self.module.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in self.datamodule.test_loader:
                images, targets = batch
                outputs = self.module(images)
                preds = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        return all_preds, all_targets
    
    def inference_with_tta(self,
                           num_tta: int = 5):
        self.module.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in self.datamodule:
                images, targets = batch
                batch_preds = []
                for _ in range(num_tta):
                    outputs = self.module(images)
                    preds = torch.softmax(outputs, dim=1).cpu().numpy()
                    batch_preds.append(preds)
                batch_preds = np.mean(batch_preds, axis=0)
                all_preds.append(batch_preds)
                all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return all_preds, all_targets
    
    def compute_metrics(preds: np.array,
                        targets: np.array):
        pred_labels = np.argmax(preds, axis=1)
        accuracy = accuracy_score(targets, pred_labels)
        auroc = roc_auc_score(targets, preds[:, 1])
        f1 = f1_score(targets, pred_labels)
        
        precision, recall, _ = precision_recall_curve(targets, preds[:, 1])
        
        # Filter precision and recall values where recall >= 0.8
        
        filtered_precision = precision[recall >= 0.8]
        filtered_recall = recall[recall >= 0.8]
        pauroc = auc(filtered_recall, filtered_precision)
        
        return {
            'accuracy': accuracy,
            'auroc': auroc,
            'f1': f1,
            'pauroc_0.8': pauroc
        }
    

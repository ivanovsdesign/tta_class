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

from functional.criterion import UANLLoss

import numpy as np

from omegaconf import DictConfig

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ISICInferencePipeline():
    def __init__(self,
                 datamodule: Callable,
                 module: Callable,
                 criterion: Callable,
                 **kwargs) -> None:
        
        self.__dict__.update(kwargs)
        self.datamodule = datamodule
        self.module = module
        self.criterion = criterion
        self.device = device

    def setup():
        pass

    def inference_without_tta(self):
        print('Inference without TTA')
        self.module.eval()
        all_preds = []
        all_targets = []
        all_uncertainties = []
        all_confidences = []

        with torch.no_grad():
            for batch in tqdm(self.datamodule.val_dataloader()):
                images, targets = batch
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.module(images)
                
                if isinstance(self.criterion, UANLLoss):
                    preds = torch.softmax(outputs[:, :-1], dim=1).detach().cpu()
                    uncertainties = outputs[:, -1].detach().cpu()
                else:
                    preds = torch.softmax(outputs, dim=1).detach().cpu()
                    uncertainties = torch.zeros_like(preds[:, 0])  # Placeholder for uncertainty
                
                confidences = preds.max(dim=1).values.detach().cpu()
                
                all_preds.append(preds.numpy())
                all_targets.append(targets.cpu().numpy())
                all_uncertainties.append(uncertainties.numpy())
                all_confidences.append(confidences.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_uncertainties = np.concatenate(all_uncertainties)
        all_confidences = np.concatenate(all_confidences)
        
        return all_preds, all_targets, all_uncertainties, all_confidences
    
    def inference_with_tta(self, num_tta: int = 5):
        print(f'Inference with TTA\nNumber of TTA: {num_tta}')
        self.module.eval()
        all_preds = []
        all_targets = []
        all_uncertainties = []
        all_confidences = []

        with torch.no_grad():
            for tta_index in tqdm(range(num_tta), desc="TTA Iterations"):
                batch_preds = []
                batch_uncertainties = []
                batch_confidences = []

                for batch in self.datamodule.tta_dataloader():
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                    outputs = self.module(images)
                    
                    if isinstance(self.criterion, UANLLoss):
                        preds = torch.softmax(outputs[:, :-1], dim=1).detach().cpu()
                        uncertainties = outputs[:, -1].detach().cpu()
                    else:
                        preds = torch.softmax(outputs, dim=1).detach().cpu()
                        uncertainties = torch.zeros_like(preds[:, 0])  # Placeholder for uncertainty
                    
                    confidences = preds.max(dim=1).values.detach().cpu()
                    
                    batch_preds.append(preds.numpy())
                    batch_uncertainties.append(uncertainties.numpy())
                    batch_confidences.append(confidences.numpy())
                    
                    if tta_index == 0:
                        all_targets.append(targets.cpu().numpy())

                all_preds.append(np.concatenate(batch_preds, axis=0))
                all_uncertainties.append(np.concatenate(batch_uncertainties, axis=0))
                all_confidences.append(np.concatenate(batch_confidences, axis=0))

        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_uncertainties = np.array(all_uncertainties)
        all_confidences = np.array(all_confidences)

        # Average the predictions, uncertainties, and confidences over the TTA iterations
        avg_preds = np.mean(all_preds, axis=0)
        avg_uncertainties = np.mean(all_uncertainties, axis=0)
        avg_confidences = np.mean(all_confidences, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        return avg_preds, all_targets, avg_uncertainties, avg_confidences
    
    def compute_metrics(self,
                        preds: np.array,
                        targets: np.array,
                        confidences: np.array,
                        certainties: np.array = None,
                        prefix: str = ''):

        if certainties is None:
            certainties = np.ones_like(confidences)  # Use confidences if certainties are not provided

        regular_metrics = self._compute_metrics(preds, targets, prefix + '_regular')

        weighted_preds_confidences = preds * confidences[:, np.newaxis]
        confidences_metrics = self._compute_metrics(weighted_preds_confidences, targets, prefix + '_confidences')

        if isinstance(self.criterion, UANLLoss):
            weighted_preds_certainties = preds * (certainties[:, np.newaxis] ** 2)
            certainties_metrics = self._compute_metrics(weighted_preds_certainties, targets, prefix + '_certainties')
        else:
            certainties_metrics = {}

        all_metrics = {**regular_metrics, **confidences_metrics, **certainties_metrics}

        return all_metrics

    def _compute_metrics(self, preds: np.array, targets: np.array, prefix: str):
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
            f'{prefix}_accuracy': np.array(accuracy, dtype=np.float64),
            f'{prefix}_auroc': np.array(auroc, dtype=np.float64),
            f'{prefix}_f1': np.array(f1, dtype=np.float64),
            f'{prefix}_pauroc_0.8': np.array(pauroc, dtype=np.float64)
        }
    

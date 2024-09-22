import torch
import torch.nn as nn
from typing import Callable

import pytorch_lightning as pl

from functional.criterion import UANLLoss

class ClassModel(pl.LightningModule):
    def __init__(self,
                 model: Callable,
                 criterion: Callable):
        super(ClassModel, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        if isinstance(self.criterion, UANLLoss):
            # Split the output into predictions and uncertainty
            predictions = outputs[:, :-1]
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        if isinstance(self.criterion, UANLLoss):
            predictions = outputs[:, :-1]
        else: 
            predictions = outputs
        
        loss = self.criterion(outputs, targets)
        
        sm = nn.functional.softmax(predictions, dim=1)
        acc = (sm.argmax(1) == targets).sum() / len(targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer
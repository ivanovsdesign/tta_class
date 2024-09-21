import torch
import torch.nn as nn
from typing import Callable

import pytorch_lightning as pl

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
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        sm = nn.functional.softmax(outputs, dim=1)
        acc = (sm.argmax(1) == targets).sum() / len(targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 16 * 50, gamma=0.5, last_epoch=-1, verbose=0)
        # return [optimizer], [scheduler]
        return optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from functional.criterion import UANLLoss

from typing import Callable

import timm

def gem(x, p=3, eps=1e-4):
    return F.avg_pool2d(x.clamp(min=eps), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps


    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret
    

class ISICModel(nn.Module):
    def __init__(self,
                 backbone: str,
                 drop_rate: float,
                 drop_rate_path: float,
                 drop_rate_last: float,
                 criterion: Callable = nn.CrossEntropyLoss,
                 pretrained: bool = False,
                 out_chans: int = 2):
        '''
        Model with adaptive output FC for using with UANLL Loss
        ! Uses external variable `LOSS`
        '''
        super(ISICModel, self).__init__()
        self.encoder = timm.create_model(
            backbone,
            in_chans=3,
            drop_rate=drop_rate,
            drop_path_rate=drop_rate_path,
            pretrained=pretrained,
        )

        self.out_chans = out_chans
        
        self.nb_fts = self.encoder.num_features
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
                        nn.Linear(self.nb_fts, 128),
                        nn.BatchNorm1d(128),
                        nn.Dropout(drop_rate_last),
                        nn.LeakyReLU(0.1),
                        nn.Linear(128, self.out_chans),
                    )
        
    def forward(self, x):
        feat = self.encoder.forward_features(x)
        feat = self.gap(feat)[:,:,0,0]
        y = self.head(feat)
        
        return y
    
    def freeze_encoder(self, flag):
        for param in self.encoder.parameters():
            param.requires_grad = not flag
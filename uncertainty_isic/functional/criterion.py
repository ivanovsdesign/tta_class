import torch
import torch.nn as nn

class UANLLoss(nn.Module):
    def __init__(self,
                 smoothing: float,
                 classes: int):
        super(UANLLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.clc = classes
        
    def forward(self,x,y):
        logvar  = (x[:,self.clc:]) #** 2 
        prob = x[:,:self.clc]

        with torch.no_grad():
            yoh = torch.zeros_like(prob)
            yoh.fill_(self.smoothing / (self.clc - 1))
            yoh.scatter_(1, y.data.unsqueeze(1), self.confidence)

        loss0 = ((yoh - prob) ** 2).sum(dim=1)
        loss = (torch.exp(-logvar) * loss0 + self.clc * logvar)

        return loss.mean()
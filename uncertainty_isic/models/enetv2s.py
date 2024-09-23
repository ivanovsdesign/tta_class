import torch.nn as nn
import timm

class ISICModel(nn.Module):
    def __init__(self,
                 backbone: str,
                 drop_rate: float,
                 drop_rate_path: float,
                 pretrained: bool = False,
                 out_chans: int = 2):
        super(ISICModel, self).__init__()
        self.encoder = timm.create_model(
            backbone,
            in_chans=3,
            drop_rate=drop_rate,
            drop_path_rate=drop_rate_path,
            pretrained=pretrained,
            num_classes=out_chans
        )

    def forward(self, x):
        return self.encoder(x)
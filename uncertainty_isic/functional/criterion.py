import torch
import torch.nn as nn

class UANLLoss(nn.Module):
    '''
    Proposed Uncertainty-aware negative log-likelihood loss (UANLL)
    Requires uncertainty estimation output of the model (so, the output dimension should be num_classes + 1)

    Label smoothing doesn't work yet

    Calculates loss based on formula:
    .. math::
        L_{C} = \frac{1}{2m}\sum_{i=1}^{m} \left(e^{-s^{(i)}}  \sum_{k=1}^{N}\left(y_k^{(i)} - h_k^{(i)}\right)^2 + N s^{(i)}\right)

    Args:
        num_classes (int): Number of classes of a dataset
        label_smoothing (float): Amount of label smoothing
    '''

    def __init__(self,
                 num_classes: int = 2,
                 label_smoothing: float = 0.) -> None:
        super(UANLLoss, self).__init__()
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.num_classes = num_classes

    def __str__(self) -> str:
        return self.__class__.__name__

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        logstd = (x[:, self.num_classes:]) ** 2
        prob = nn.functional.softmax(x[:, :self.num_classes], 1)

        with torch.no_grad():
            yoh = torch.zeros_like(prob)
            yoh.fill_(self.smoothing / (self.num_classes - 1))
            yoh.scatter_(1, y.data.to(
                torch.int64).unsqueeze(1), self.confidence)

        loss0 = (yoh - prob) ** 2
        loss = (torch.exp(-logstd) * loss0.sum(dim=1) +
                self.num_classes * logstd)
        return loss.mean()
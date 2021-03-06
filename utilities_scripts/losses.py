import torch
import torch.nn.functional as F
import torch.nn as nn


def get_criterion(smooth=0.0, multilabel=False):
    if smooth > 0:
        if multilabel == False:
            criterion = LabelSmoothing(smooth)
        else:
            criterion = NLLMultiLabelSmooth(smooth)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
import torch


class BinaryIoU:
    """Tiny replacement for torchmetrics.JaccardIndex."""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.i = 0
        self.u = 0

    def update(self, preds, targets, thresh=0.5):
        """Update the metric with new predictions and targets."""
        preds = (preds > thresh).bool()
        self.i += torch.logical_and(preds, targets).sum().item()
        self.u += torch.logical_or(preds, targets).sum().item()

    def compute(self):
        """Compute the current value of the metric."""
        return self.i / (self.u + self.eps)

    def reset(self):
        """Reset the metric to its initial state."""
        self.i = self.u = 0

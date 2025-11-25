import torch
from torch import nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predictions, targets):
        return nn.MSELoss()(predictions, targets)

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        ce_loss = nn.CrossEntropyLoss()(predictions, targets)
        focal_loss = self.alpha * (1 - predictions) ** self.gamma * ce_loss
        return focal_loss.mean()

class FocalPoissonNLLLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalPoissonNLLLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, data_batch):
        # Compute the Poisson negative log-likelihood with node masking
        batch_size = predictions.shape[0]
        targets = data_batch['targets'][:, :-1]  # Shape: (batch, nodes, seq_len)
        node_mask = data_batch['node_mask'][:, :-1]  # Shape: (batch, nodes)

        # Expand mask to match sequence length: (batch, nodes, seq_len)
        node_mask_expanded = node_mask.unsqueeze(-1).expand_as(targets)

        # Transpose for loss computation: (batch, seq_len, nodes)
        node_mask_transposed = node_mask_expanded.transpose(-1, -2)

        # Compute element-wise Poisson NLL loss with reduction='none'
        nll = F.poisson_nll_loss(
            predictions.transpose(-1,-2),
            targets.transpose(-1,-2),
            log_input=True,
            reduction='none')  # Shape: (batch, seq_len, nodes)

        # Apply node mask to exclude zero nodes from loss
        masked_nll = nll * node_mask_transposed

        # Apply focal loss weighting only to masked regions
        focal_loss = self.alpha * (1 - predictions.transpose(-1,-2)) ** self.gamma * masked_nll

        # Normalize by number of active (masked) elements instead of all elements
        num_active = node_mask_transposed.sum() or 1
        return focal_loss.sum() / num_active
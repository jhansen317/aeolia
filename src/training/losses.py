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
    def __init__(self, alpha=1.0, gamma=2.0, beta=0.0):
        """
        Focal Poisson NLL Loss with optional VAE KL divergence.

        Args:
            alpha: Focal loss weight
            gamma: Focal loss exponent
            beta: Weight for KL divergence term (annealed during training)
        """
        super(FocalPoissonNLLLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta  # KL weight (will be updated during training)

    def forward(self, model_output, data_batch):
        """
        Compute loss, handling both VAE and non-VAE outputs.

        Args:
            model_output: Either (predictions, mu, logvar) tuple or just predictions
            data_batch: Batch dict with targets and node_mask

        Returns:
            Total loss (reconstruction + beta * KL if VAE)
        """
        # Unpack model output
        if isinstance(model_output, tuple):
            predictions, mu, logvar = model_output
            use_vae = True
        else:
            predictions = model_output
            use_vae = False

        # Compute reconstruction loss (Focal Poisson NLL)
        batch_size = predictions.shape[0]
        # Note: No longer need to exclude composer node (already removed from graph)
        targets = data_batch['targets']  # Shape: (batch, nodes, seq_len)
        node_mask = data_batch['node_mask']  # Shape: (batch, nodes)

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
        recon_loss = focal_loss.sum() / num_active

        # Add KL divergence if using VAE
        if use_vae and self.beta > 0:
            # KL(q(z|x) || p(z)) where p(z) = N(0, I)
            # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # Shape of mu/logvar: (batch, bottleneck_nodes, channels, timesteps)
            # KL is computed at bottleneck (latent space), no masking needed
            kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

            # Average over all latent dimensions (batch, nodes, channels, timesteps)
            kl_loss = kl_per_element.mean()

            total_loss = recon_loss + self.beta * kl_loss
            return total_loss
        else:
            return recon_loss

    def set_beta(self, beta):
        """Update KL weight (for annealing)."""
        self.beta = beta
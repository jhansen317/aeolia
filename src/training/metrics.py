"""
Evaluation metrics for music generation.

These metrics handle the model's log-probability outputs and compare them
against ground truth targets.
"""

import torch
import torch.nn.functional as F

# Global threshold for note on/off detection
NOTE_THRESHOLD = 0.01


class MusicGenerationMetrics:
    """
    Metrics for evaluating music generation quality.

    Handles models that output log probabilities (from log_softmax or similar)
    and compares against count-based targets (e.g., velocities).
    """

    def __init__(self, threshold=None):
        """
        Args:
            threshold: Velocity threshold for considering a note "on" (default: NOTE_THRESHOLD)
        """
        self.threshold = threshold if threshold is not None else NOTE_THRESHOLD

    def __call__(self, predictions, targets, node_mask=None):
        """
        Calculate all metrics.

        Args:
            predictions: Model outputs (log probabilities), shape (batch, nodes, time)
            targets: Ground truth values, shape (batch, nodes, time)
            node_mask: Optional mask for active nodes, shape (batch, nodes)

        Returns:
            dict: Dictionary of metric values
        """
        # Convert log probabilities to probabilities
        if predictions.max() <= 0:  # Likely log probs
            predictions = predictions.exp()

        return self.calculate_metrics(predictions, targets, node_mask)

    def calculate_metrics(self, predictions, targets, node_mask=None):
        """
        Calculate precision, recall, F1, and accuracy.

        Args:
            predictions: Probabilities (NOT log probs), shape (batch, nodes, time)
            targets: Ground truth values, shape (batch, nodes, time)
            node_mask: Optional mask for active nodes, shape (batch, nodes)

        Returns:
            dict with keys: precision, recall, f1, note_accuracy, mse
        """
        # Ensure tensors are on same device
        device = predictions.device
        targets = targets.to(device)
        if node_mask is not None:
            node_mask = node_mask.to(device)

        # Convert to binary (note on/off) based on threshold
        pred_binary = (predictions > self.threshold).float()
        target_binary = (targets > self.threshold).float()

        # Apply node mask if provided (exclude padding/unused nodes)
        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(-1).expand_as(pred_binary)

            # Calculate true positives, false positives, false negatives, true negatives
            # Only on masked (active) nodes
            tp = (pred_binary * target_binary * mask_expanded).sum()
            fp = (pred_binary * (1 - target_binary) * mask_expanded).sum()
            fn = ((1 - pred_binary) * target_binary * mask_expanded).sum()
            tn = ((1 - pred_binary) * (1 - target_binary) * mask_expanded).sum()

            # Calculate MSE only on active nodes
            pred_masked = predictions * mask_expanded
            target_masked = targets * mask_expanded
            mse = (pred_masked - target_masked).pow(2).sum() / mask_expanded.sum()
        else:
            # Calculate true positives, false positives, false negatives, true negatives
            tp = (pred_binary * target_binary).sum()
            fp = (pred_binary * (1 - target_binary)).sum()
            fn = ((1 - pred_binary) * target_binary).sum()
            tn = ((1 - pred_binary) * (1 - target_binary)).sum()

            # Calculate MSE for continuous values
            mse = F.mse_loss(predictions, targets)

        # Calculate metrics with small epsilon to avoid division by zero
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'note_accuracy': accuracy.item(),
            'mse': mse.item(),
            'tp': tp.item(),
            'fp': fp.item(),
            'fn': fn.item(),
            'tn': tn.item()
        }

    def calculate_per_sample_metrics(self, predictions, targets, node_mask=None):
        """
        Calculate metrics for each sample in the batch separately.

        Args:
            predictions: Probabilities, shape (batch, nodes, time)
            targets: Ground truth, shape (batch, nodes, time)
            node_mask: Optional mask, shape (batch, nodes)

        Returns:
            List of dicts, one per sample
        """
        batch_size = predictions.shape[0]
        metrics_list = []

        for i in range(batch_size):
            pred_i = predictions[i:i+1]
            target_i = targets[i:i+1]
            mask_i = node_mask[i:i+1] if node_mask is not None else None

            metrics = self.calculate_metrics(pred_i, target_i, mask_i)
            metrics_list.append(metrics)

        return metrics_list


class NoteOnsetMetrics:
    """
    Metrics specifically for note onset detection (when notes start).

    More lenient than frame-level metrics - considers a note detected if
    it's predicted within a small time window of the actual onset.
    """

    def __init__(self, tolerance_frames=1, threshold=None):
        """
        Args:
            tolerance_frames: Number of frames of tolerance for onset timing
            threshold: Threshold for considering a note "on" (default: NOTE_THRESHOLD)
        """
        self.tolerance = tolerance_frames
        self.threshold = threshold if threshold is not None else NOTE_THRESHOLD

    def __call__(self, predictions, targets, node_mask=None):
        """
        Calculate onset-based metrics.

        Args:
            predictions: Probabilities, shape (batch, nodes, time)
            targets: Ground truth, shape (batch, nodes, time)
            node_mask: Optional mask, shape (batch, nodes)

        Returns:
            dict: onset_precision, onset_recall, onset_f1
        """
        # Convert to binary
        pred_binary = (predictions > self.threshold).float()
        target_binary = (targets > self.threshold).float()

        # Detect onsets (transitions from 0 to 1)
        pred_onsets = self._detect_onsets(pred_binary)
        target_onsets = self._detect_onsets(target_binary)

        # Apply node mask if provided
        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(-1).expand_as(pred_onsets)
            pred_onsets = pred_onsets * mask_expanded
            target_onsets = target_onsets * mask_expanded

        # Match onsets within tolerance
        tp = self._count_matched_onsets(pred_onsets, target_onsets, node_mask)
        fp = pred_onsets.sum() - tp
        fn = target_onsets.sum() - tp

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        return {
            'onset_precision': precision.item(),
            'onset_recall': recall.item(),
            'onset_f1': f1.item()
        }

    def _detect_onsets(self, binary_sequence):
        """Detect note onsets (0 -> 1 transitions)."""
        # Pad with zeros at the start
        padded = F.pad(binary_sequence, (1, 0), value=0)

        # Onset is where current frame is 1 and previous was 0
        onsets = binary_sequence * (1 - padded[:, :, :-1])
        return onsets

    def _count_matched_onsets(self, pred_onsets, target_onsets, node_mask=None):
        """Count how many predicted onsets match targets within tolerance."""
        # This is a simplified version - could be improved with proper onset matching
        # For now, just check if onsets align within tolerance window
        batch, nodes, time = pred_onsets.shape

        matched = 0
        for b in range(batch):
            for n in range(nodes):
                # Skip masked (inactive) nodes
                if node_mask is not None and not node_mask[b, n]:
                    continue

                pred_times = torch.where(pred_onsets[b, n] > 0)[0]
                target_times = torch.where(target_onsets[b, n] > 0)[0]

                for pt in pred_times:
                    # Check if any target onset is within tolerance
                    if len(target_times) > 0:
                        distances = torch.abs(target_times - pt)
                        if distances.min() <= self.tolerance:
                            matched += 1

        return torch.tensor(matched, dtype=torch.float32)

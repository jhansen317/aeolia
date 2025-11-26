"""
TensorBoard logging utilities for training visualization.

This module provides a clean interface for logging metrics, images, and
other training artifacts to TensorBoard.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List, Union
import io
from PIL import Image


class TensorBoardLogger:
    """
    Wrapper for TensorBoard SummaryWriter with convenient methods for
    logging training metrics and visualizations.
    """

    def __init__(self, log_dir: Union[str, Path], comment: str = ''):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            comment: Optional comment to append to run name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)
        self.global_step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value.

        Args:
            tag: Name of the scalar (e.g., 'loss/train', 'metrics/accuracy')
            value: Scalar value to log
            step: Global step (defaults to internal counter)
        """
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, value_dict: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple scalar values in one chart.

        Args:
            tag: Main tag for the chart
            value_dict: Dictionary of {sub_tag: value}
            step: Global step (defaults to internal counter)
        """
        step = step if step is not None else self.global_step
        self.writer.add_scalars(tag, value_dict, step)

    def log_metrics_dict(self, metrics: Dict[str, float], prefix: str = '', step: Optional[int] = None):
        """
        Log a dictionary of metrics as individual scalars.

        Args:
            metrics: Dictionary of metric names to values
            prefix: Prefix to add to all metric names (e.g., 'train/', 'val/')
            step: Global step (defaults to internal counter)
        """
        step = step if step is not None else self.global_step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{prefix}{key}", value, step)

    def log_figure(self, tag: str, figure: plt.Figure, step: Optional[int] = None, close: bool = True):
        """
        Log a matplotlib figure as an image.

        Args:
            tag: Name for the image
            figure: Matplotlib figure object
            step: Global step (defaults to internal counter)
            close: Whether to close the figure after logging
        """
        step = step if step is not None else self.global_step

        # Convert figure to image
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)

        # Convert to tensor (C, H, W)
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)

        self.writer.add_image(tag, image_tensor, step)
        buf.close()

        if close:
            plt.close(figure)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """
        Log an image tensor or array.

        Args:
            tag: Name for the image
            image: Image as tensor (C, H, W) or array
            step: Global step (defaults to internal counter)
        """
        step = step if step is not None else self.global_step

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure correct shape (C, H, W)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[0] not in [1, 3, 4]:
            # Assume (H, W, C) and convert to (C, H, W)
            image = image.permute(2, 0, 1)

        self.writer.add_image(tag, image, step)

    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """
        Log a batch of images.

        Args:
            tag: Name for the images
            images: Images as tensor (N, C, H, W) or array
            step: Global step (defaults to internal counter)
        """
        step = step if step is not None else self.global_step

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        self.writer.add_images(tag, images, step)

    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """
        Log a histogram of values.

        Args:
            tag: Name for the histogram
            values: Values to create histogram from
            step: Global step (defaults to internal counter)
        """
        step = step if step is not None else self.global_step

        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)

        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """
        Log text data.

        Args:
            tag: Name for the text
            text: Text content
            step: Global step (defaults to internal counter)
        """
        step = step if step is not None else self.global_step
        self.writer.add_text(tag, text, step)

    def log_model_graph(self, model: torch.nn.Module, input_data):
        """
        Log the model computational graph.

        Args:
            model: PyTorch model
            input_data: Example input to the model
        """
        self.writer.add_graph(model, input_data)

    def log_hyperparameters(self, hparam_dict: Dict, metric_dict: Dict):
        """
        Log hyperparameters and metrics for comparison.

        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of metric values
        """
        self.writer.add_hparams(hparam_dict, metric_dict)

    def increment_step(self):
        """Increment the global step counter."""
        self.global_step += 1

    def set_step(self, step: int):
        """Set the global step counter to a specific value."""
        self.global_step = step

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def figure_to_tensor(figure: plt.Figure) -> torch.Tensor:
    """
    Convert a matplotlib figure to a tensor.

    Args:
        figure: Matplotlib figure

    Returns:
        Tensor of shape (C, H, W)
    """
    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    tensor = torch.tensor(np.array(image)).permute(2, 0, 1)
    buf.close()
    return tensor

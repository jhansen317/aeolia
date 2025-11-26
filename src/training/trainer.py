#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from src.data.dataset import batch_to_device
from src.models.astgcn import PolyphonyGCN
from src.training.metrics import MusicGenerationMetrics, NoteOnsetMetrics
from configs.default_config import DefaultConfig
from src.utils.visualization import (
    visualize_predictions,
    visualize_graph_structure,
    visualize_activations
)
from src.utils.tensorboard_logger import TensorBoardLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# torch.autograd.set_detect_anomaly(True)  # Disabled for performance - only enable for debugging
if os.environ.get('DEBUG_AUTOGRAD', '0') == '1':
    torch.autograd.set_detect_anomaly(True)
    logger.warning("Autograd anomaly detection enabled - training will be ~10x slower")

activations = {}
inputs = {}


def save_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach().cpu()
        inputs[name] = input[0].detach().cpu()
    return hook


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, composer_criterion, device, config, tb_logger=None):
        self.model = model
        model.encoder[0].register_forward_hook(save_activation('encoderlayer0'))
        model.encoder[1].register_forward_hook(save_activation('encoderlayer1'))
        model.encoder[2].register_forward_hook(save_activation('encoderlayer2'))
        model.decoder[2].register_forward_hook(save_activation('decoderlayer0'))
        model.decoder[1].register_forward_hook(save_activation('decoderlayer1'))
        model.decoder[0].register_forward_hook(save_activation('decoderlayer2'))

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.composer_criterion = composer_criterion
        self.device = device
        self.config = config

        # Initialize TensorBoard logger
        if tb_logger is None:
            log_dir = project_root / "runs"
            self.tb_logger = TensorBoardLogger(log_dir)
        else:
            self.tb_logger = tb_logger

        # Initialize metrics
        self.metrics = MusicGenerationMetrics()
        self.onset_metrics = NoteOnsetMetrics(tolerance_frames=1)

        self.checkpoint_dir = project_root / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create output directory for visualizations (for debugging only)
        self.vis_dir = project_root / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)

        # Training state
        self.best_loss = float('inf')
        self.global_step = 0

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': vars(self.config)
        }
        filename = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, filename)
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model with loss {loss:.4f}")

    def load_checkpoint(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Loaded checkpoint from {path}, epoch {checkpoint['epoch']}, loss {checkpoint['loss']}")
        return checkpoint['epoch'], checkpoint['loss']

    def log_visualizations(self, epoch, example_idx, data, output):
        """Log all visualizations to TensorBoard."""

        # Prepare data
        model_input = data.get('features') if isinstance(data, dict) else getattr(data, 'features', None)
        model_output_probs = output[:, :233, :].exp()  # Convert to probs and slice to match
        node_mask_vis = data.get('node_mask') if isinstance(data, dict) else getattr(data, 'node_mask', None)
        if node_mask_vis is not None:
            node_mask_vis = node_mask_vis[:, :233]

        # 1. Visualize activations
        fig_activations = visualize_activations(
            activations,
            inputs,
            self.model.num_layers,
            model_input=model_input,
            model_output=model_output_probs,
            node_mask=node_mask_vis,
            save_path=None  # Don't save to file, return figure
        )
        if fig_activations is not None:
            self.tb_logger.log_figure('visualizations/activations', fig_activations, self.global_step)

        # 2. Visualize predictions
        if hasattr(data, 'file_paths') or 'file_paths' in data:
            try:
                file_paths = data.get('file_paths') if isinstance(data, dict) else data.file_paths
                targets = data.get('targets') if isinstance(data, dict) else data.targets
                targets = targets[:, :233, :]
                node_mask = data.get('node_mask') if isinstance(data, dict) else getattr(data, 'node_mask', None)
                node_mask = node_mask[:, :233]

                # Slice output and convert to probabilities
                output_sliced = output[:, :233, :].exp()

                # Calculate per-sample metrics for visualization
                per_sample_metrics = self.metrics.calculate_per_sample_metrics(
                    output_sliced, targets, node_mask
                )

                fig_predictions = visualize_predictions(
                    output_sliced,
                    targets,
                    filenames=file_paths,
                    metrics_list=per_sample_metrics,
                    save_path=None  # Don't save to file, return figure
                )
                if fig_predictions is not None:
                    self.tb_logger.log_figure('visualizations/predictions', fig_predictions, self.global_step)
            except Exception as e:
                logger.warning(f"Could not visualize predictions: {e}")

        # 3. Visualize graph structure
        if hasattr(data, 'edge_index') or 'input_graphs' in data or 'global_graph' in data:
            try:
                # Use global_graph if available (when use_global_graph=True), otherwise first per-timestep graph
                if isinstance(data, dict):
                    if 'global_graph' in data and data['global_graph'] is not None:
                        graph_data = data['global_graph']
                    elif 'input_graphs' in data and len(data['input_graphs']) > 0:
                        graph_data = data['input_graphs'][0]
                    else:
                        graph_data = None
                else:
                    graph_data = data

                if graph_data is not None:
                    fig_graph = visualize_graph_structure(
                        graph_data,
                        save_path=None,  # Don't save to file, return figure
                        render_3d=True
                    )
                    if fig_graph is not None:
                        self.tb_logger.log_figure('visualizations/graph_structure', fig_graph, self.global_step)
            except Exception as e:
                logger.warning(f"Could not visualize graph structure: {e}")

    def log_batch_metrics(self, epoch, example_idx, loss, data, output):
        """Log metrics for a single batch to TensorBoard."""

        # Log losses
        self.tb_logger.log_scalar('loss/batch', loss.item(), self.global_step)

        # Calculate and log detailed metrics
        if hasattr(data, 'file_paths') or 'file_paths' in data:
            try:
                targets = data.get('targets') if isinstance(data, dict) else data.targets
                targets = targets[:, :233, :]
                node_mask = data.get('node_mask') if isinstance(data, dict) else getattr(data, 'node_mask', None)
                node_mask = node_mask[:, :233]

                # Slice output to match targets and node_mask dimensions
                output_sliced = output[:, :233, :].exp()

                # Calculate all metrics (frame-level and onset-based)
                metrics_dict = self.metrics.calculate_metrics(output_sliced, targets, node_mask)
                onset_dict = self.onset_metrics(output_sliced, targets, node_mask)
                metrics_dict.update(onset_dict)

                # Log all metrics to TensorBoard
                self.tb_logger.log_metrics_dict(metrics_dict, prefix='metrics/', step=self.global_step)

                # Log a concise summary to console
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} | "
                    f"Batch {example_idx} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {metrics_dict['note_accuracy']:.3f} | "
                    f"F1: {metrics_dict['f1']:.3f} | "
                    f"Onset F1: {metrics_dict['onset_f1']:.3f}"
                )
            except Exception as e:
                logger.warning(f"Could not calculate metrics: {e}")

    def log_epoch_summary(self, epoch, avg_loss, current_lr):
        """Log epoch-level summary statistics."""
        self.tb_logger.log_scalar('loss/epoch', avg_loss, epoch)
        self.tb_logger.log_scalar('learning_rate', current_lr, epoch)

        logger.info(
            f"\n{'='*60}\n"
            f"Epoch {epoch+1}/{self.config.num_epochs} Summary\n"
            f"Average Loss: {avg_loss:.4f}\n"
            f"Best Loss: {self.best_loss:.4f}\n"
            f"Learning Rate: {current_lr:.6f}\n"
            f"{'='*60}\n"
        )

    def train(self, train_loader, epochs):
        """
        Main training loop with clean TensorBoard logging.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for example_idx, data in enumerate(train_loader):
                # Move batch to device
                data = batch_to_device(data, self.device)

                # Forward pass
                output = self.model(data)

                # Check for NaN/Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error("NaN or Inf detected in model output!")
                    continue

                # Calculate loss
                loss = self.criterion(output, data)

                # Log batch metrics
                self.log_batch_metrics(epoch, example_idx, loss, data, output)

                # Visualizations (every 5 batches to avoid overhead)
                if (example_idx % 10) == 0:
                    self.log_visualizations(epoch, example_idx, data, output)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # Track total loss
                total_loss += loss.item()
                avg_loss = total_loss / (example_idx + 1)

                # Update best loss and save checkpoint
                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
                    self.save_checkpoint(epoch, avg_loss, is_best=True)
                elif (example_idx % 5) == 0:
                    self.save_checkpoint(epoch, avg_loss, is_best=False)

                # Increment global step
                self.global_step += 1

                # Clean up matplotlib figures
                plt.close('all')

            # Log epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            self.log_epoch_summary(epoch, avg_loss, current_lr)

        logger.info("Training complete!")
        self.tb_logger.close()

    def validate(self, val_loader):
        """
        Validation loop with TensorBoard logging.

        Args:
            val_loader: DataLoader for validation data
        """
        self.model.eval()
        total_loss = 0
        all_metrics = []

        with torch.no_grad():
            for data in val_loader:
                data = batch_to_device(data, self.device)
                output = self.model(data)
                loss = self.criterion(output, data)
                total_loss += loss.item()

                # Calculate metrics
                try:
                    targets = data.get('targets') if isinstance(data, dict) else data.targets
                    targets = targets[:, :233, :]
                    node_mask = data.get('node_mask') if isinstance(data, dict) else getattr(data, 'node_mask', None)
                    node_mask = node_mask[:, :233]
                    output_sliced = output[:, :233, :].exp()

                    metrics_dict = self.metrics.calculate_metrics(output_sliced, targets, node_mask)
                    onset_dict = self.onset_metrics(output_sliced, targets, node_mask)
                    metrics_dict.update(onset_dict)
                    all_metrics.append(metrics_dict)
                except Exception as e:
                    logger.warning(f"Could not calculate validation metrics: {e}")

        # Average metrics across all batches
        avg_loss = total_loss / len(val_loader)

        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

            # Log validation metrics
            self.tb_logger.log_scalar('loss/validation', avg_loss, self.global_step)
            self.tb_logger.log_metrics_dict(avg_metrics, prefix='val_metrics/', step=self.global_step)

            logger.info(
                f"Validation - Loss: {avg_loss:.4f} | "
                f"Acc: {avg_metrics['note_accuracy']:.3f} | "
                f"F1: {avg_metrics['f1']:.3f}"
            )
        else:
            logger.info(f"Validation Loss: {avg_loss:.4f}")

        return avg_loss

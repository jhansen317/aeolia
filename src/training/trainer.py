#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt





# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
#from torch_geometric_temporal.signal import temporal_signal_split
from src.data.dataset import MidiGraphDataset, temporal_graph_collate, batch_to_device
from src.models.astgcn import PolyphonyGCN
from src.training.metrics import MusicGenerationMetrics, NoteOnsetMetrics
from configs.default_config import DefaultConfig
from src.utils.visualization import (
    visualize_predictions,
    visualize_graph_structure,
    visualize_activations
)

torch.autograd.set_detect_anomaly(True)

activations = {}
inputs = {}
def save_activation(name):
    def hook(model, input, output):
        activations[name] = output[0].detach().cpu()
        inputs[name] = input[0].detach().cpu()
    return hook



class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, composer_criterion, device, config):
        self.model = model
        model.encoder[0].register_forward_hook(save_activation('encoderlayer0'))
        model.encoder[1].register_forward_hook(save_activation('encoderlayer1'))
        model.encoder[2].register_forward_hook(save_activation('encoderlayer2'))
        model.decoder[0].register_forward_hook(save_activation('decoderlayer0'))
        model.decoder[1].register_forward_hook(save_activation('decoderlayer1'))
        model.decoder[2].register_forward_hook(save_activation('decoderlayer2'))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.composer_criterion = composer_criterion
        self.device = device
        self.config = config

        # Initialize metrics
        self.metrics = MusicGenerationMetrics(threshold=0.5)
        self.onset_metrics = NoteOnsetMetrics(tolerance_frames=1)
        data_dir = project_root / "data" / "raw_test"
        self.checkpoint_dir = project_root / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create output directory for visualizations
        self.vis_dir = project_root / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)

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

    def load_checkpoint(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {path}, epoch {checkpoint['epoch']}, loss {checkpoint['loss']}")
        return checkpoint['epoch'], checkpoint['loss']
    
    def train(self, train_loader, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for example_idx, data in enumerate(train_loader):
                # Move batch to device
                data = batch_to_device(data, self.device)
                # Forward pass
                output = self.model(data)
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print("NaN or Inf detected in model output!")
                print(f'Output shape: {output.shape}')
                loss = self.criterion(output, data)

                #composer_loss = self.composer_criterion(composer_out, torch.tensor(data['composer_id']).to(self.device))
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print("NaN or Inf detected in model output!")
                #composer_loss = self.composer_criterion(composer_out, torch.tensor(data['composer_id']))

                #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                example_loss = loss
                #example_loss += composer_loss
                #print(f'composer loss: {composer_loss}')
                print(f' loss: {loss}')
                #example_loss += (kl_loss.mean()*0.001)

                # Visualizations (every N examples to avoid overhead)
                if (example_idx % 5) == 0:
                    # 1. Visualize all layer activations in one figure
                    visualize_activations(
                        activations,
                        inputs,
                        self.model.num_layers,
                        save_path=self.vis_dir / f"activations_epoch{epoch}_ex{example_idx}.png"
                    )

                    # 2. Calculate metrics and visualize predictions
                    if hasattr(data, 'file_paths') or 'file_paths' in data:
                        try:
                            file_paths = data.get('file_paths') if isinstance(data, dict) else data.file_paths
                            targets = data.get('targets') if isinstance(data, dict) else data.targets
                            node_mask = data.get('node_mask') if isinstance(data, dict) else getattr(data, 'node_mask', None)

                            # Calculate metrics using proper metrics module
                            metrics_dict = self.metrics(output, targets, node_mask)

                            # Also calculate onset metrics
                            onset_dict = self.onset_metrics(output.exp(), targets, node_mask)
                            metrics_dict.update(onset_dict)

                            # Calculate per-sample metrics for visualization
                            per_sample_metrics = self.metrics.calculate_per_sample_metrics(
                                output.exp(), targets, node_mask
                            )

                            # Visualize predictions with metrics
                            visualize_predictions(
                                output.exp(),
                                targets,
                                filenames=file_paths,
                                metrics_list=per_sample_metrics,
                                save_path=self.vis_dir / f"predictions_epoch{epoch}_ex{example_idx}.png"
                            )

                            print(f"Prediction metrics: Acc={metrics_dict['note_accuracy']:.3f}, "
                                  f"Precision={metrics_dict['precision']:.3f}, Recall={metrics_dict['recall']:.3f}, "
                                  f"F1={metrics_dict['f1']:.3f}, MSE={metrics_dict['mse']:.4f}")
                            print(f"  Onset: P={metrics_dict['onset_precision']:.3f}, R={metrics_dict['onset_recall']:.3f}, F1={metrics_dict['onset_f1']:.3f}")
                            print(f"  TP={int(metrics_dict['tp'])}, FP={int(metrics_dict['fp'])}, FN={int(metrics_dict['fn'])}, TN={int(metrics_dict['tn'])}")
                        except Exception as e:
                            print(f"Warning: Could not calculate metrics - {e}")
                            import traceback
                            traceback.print_exc()

                    # 3. Visualize graph structure
                    # Assuming data contains graph structure (edge_index, etc.)
                    if hasattr(data, 'edge_index') or 'input_graphs' in data:
                        try:
                            # Use first graph from input_graphs if available
                            graph_data = data.get('input_graphs', [None])[0] if isinstance(data, dict) else data
                            if graph_data is not None:
                                visualize_graph_structure(
                                    graph_data,
                                    save_path=self.vis_dir / f"graph_epoch{epoch}_ex{example_idx}.png",
                                    render_3d=True
                                )
                        except Exception as e:
                            print(f"Warning: Could not visualize graph structure - {e}")

                # Backward pass and optimization

                self.optimizer.zero_grad()
                example_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += example_loss.item()
                avg_loss = total_loss / (example_idx + 1)

                # Update best loss and save checkpoint
                if example_loss.item() < best_loss:
                    best_loss = example_loss.item()
                    self.save_checkpoint(epoch, avg_loss, is_best=True)
                elif (example_idx % 5) == 0:
                    self.save_checkpoint(epoch, avg_loss, is_best=False)
                current_lr = self.optimizer.param_groups[0]['lr']
                print("Current learning rate:", current_lr)
                print(f'Example {example_idx}, Epoch {epoch+1}/{epochs}, Example midi Loss: {loss}, classifier loss: {0.0} Avg Loss: {avg_loss}, best: {best_loss}')
                plt.close('all')

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = batch_to_device(data, self.device)
                output = self.model(data)
                loss = self.criterion(output, data)
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss/len(val_loader)}')
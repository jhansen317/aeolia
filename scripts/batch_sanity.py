#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path





# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from torch_geometric_temporal.signal import temporal_signal_split
from src.data.dataset import MidiGraphDataset, temporal_graph_collate
from src.models.astgcn import PolyphonyGCN
from configs.default_config import DefaultConfig
from src.utils.visualization import (
    visualize_graph_at,
    visualize_batch_piano_rolls,
    visualize_detailed_piano_roll,
    visualize_batch_output_piano_rolls,
    visualize_adjacency_matrix
)



def main():
    # Set up data directory
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--config', type=str, default='/Users/hansen/dev/aeolia/configs/config.yml', help='Path to the config file.')
    args = parser.parse_args()
    data_dir = project_root / "data" / "raw_test"
    
    # Create output directory for visualizations
    vis_dir = project_root / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    # Load configuration
    config = DefaultConfig(project_root=project_root, config_path=args.config)
    print("Loading dataset...")
    dataset = MidiGraphDataset(
        npz_dir=data_dir,
        seq_length=config.periods,
        time_step=config.time_step,
        max_pitch=128
    )
    
    if len(dataset) == 0:
        print("Error: No data found in dataset!")
        return
    
    print(f"Dataset contains {len(dataset)} segments")
    
    # Use a small batch size to limit memory usage
    batch_size = config.batch_size if config.batch_size <= len(dataset) else len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_graph_collate)
    model = PolyphonyGCN(config)
    
    # Use a small batch size to limit memory usage
    batch_size = min(8, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_graph_collate)
    batch = next(iter(dataloader))
    outputs = model(batch)
    visualize_graph_at(batch, time=0, save_path=vis_dir / "graph_at_time_0test.png")
    

if __name__ == "__main__":
    main()
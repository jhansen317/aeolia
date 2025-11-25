import argparse
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR



# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.models.astgcn import PolyphonyGCN
from src.training.trainer import Trainer
from src.training.losses import FocalPoissonNLLLoss, FocalCrossEntropyLoss
from configs.default_config import DefaultConfig
from src.data.dataset import MidiGraphDataset, temporal_graph_collate



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--config', type=str, default='/Users/jakobhansen/dev/aeolia/configs/config.yml', help='Path to the config file.')
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
        max_pitch=127,
        config=config
    )
    
    if len(dataset) == 0:
        print("Error: No data found in dataset!")
        return
    
    print(f"Dataset contains {len(dataset)} segments")

    # Use a small batch size to limit memory usage
    batch_size = config.batch_size if config.batch_size <= len(dataset) else len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_graph_collate)

    # Create model and move to device
    model = PolyphonyGCN(config)
    model = model.to(config.device)
    print(f"Model moved to device: {config.device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = FocalPoissonNLLLoss(alpha=1.0, gamma=2.0)
    composer_criterion = FocalCrossEntropyLoss(alpha=1.0, gamma=2.0)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100),
        CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    ], milestones=[100])

    # Initialize the Trainer
    trainer = Trainer(model, optimizer, scheduler, criterion, composer_criterion, config.device, config)

    # Start training
    trainer.train(dataloader, config.num_epochs)

if __name__ == '__main__':
    main()
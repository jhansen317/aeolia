import argparse
import sys
import torch
import logging
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from datetime import datetime

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.models.astgcn import PolyphonyGCN
from src.training.trainer import Trainer
from src.training.losses import FocalPoissonNLLLoss, FocalCrossEntropyLoss
from configs.default_config import DefaultConfig
from src.data.dataset import MidiGraphDataset, temporal_graph_collate
from src.utils.tensorboard_logger import TensorBoardLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the ASTGCN model for music generation.')
    parser.add_argument('--config', type=str,
                        default=str(project_root / 'configs' / 'config.yml'),
                        help='Path to the config file')
    parser.add_argument('--data_dir', type=str,
                        default=str(project_root / 'data' / 'raw_test'),
                        help='Path to the data directory')
    parser.add_argument('--run_name', type=str,
                        default=None,
                        help='Optional name for this training run (for TensorBoard)')
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--start_epoch', type=int,
                        default=0,
                        help='Starting epoch (overrides checkpoint epoch if provided)')
    args = parser.parse_args()

    # Load configuration
    config = DefaultConfig(project_root=project_root, config_path=args.config)

    logger.info("=" * 60)
    logger.info("Starting ASTGCN Music Generation Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")

    # Load dataset with train/val splits
    logger.info("Loading training dataset...")
    train_dataset = MidiGraphDataset(
        npz_dir=Path(args.data_dir),
        seq_length=config.periods,
        time_step=config.time_step,
        max_pitch=127,
        config=config,
        split='train',
        val_split=getattr(config, 'val_split', 0.1),
        seed=config.seed
    )

    logger.info("Loading validation dataset...")
    val_dataset = MidiGraphDataset(
        npz_dir=Path(args.data_dir),
        seq_length=config.periods,
        time_step=config.time_step,
        max_pitch=127,
        config=config,
        split='val',
        val_split=getattr(config, 'val_split', 0.1),
        seed=config.seed
    )

    if len(train_dataset) == 0:
        logger.error("No training data found in dataset!")
        return

    if len(val_dataset) == 0:
        logger.warning("No validation data found in dataset!")

    logger.info(f"Train dataset: {len(train_dataset)} segments")
    logger.info(f"Val dataset: {len(val_dataset)} segments")

    # Create dataloaders with config-aware collate function
    batch_size = config.batch_size if config.batch_size <= len(train_dataset) else len(train_dataset)
    voice_dropout_rate = getattr(config, 'voice_dropout_rate', 0.0)
    collate_fn = partial(
        temporal_graph_collate,
        use_global_graph=config.use_global_graph,
        config=config,
        voice_dropout_rate=voice_dropout_rate
    )

    # Determine pin_memory based on device (MPS doesn't support it)
    pin_memory = True # config.pin_memory and config.device != 'mps'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0
    )
    logger.info(f"Train DataLoader: batch size: {batch_size}, num_workers: {config.num_workers}, pin_memory: {pin_memory}")
    logger.info(f"Val DataLoader: batch size: {batch_size}, num_workers: {config.num_workers}, pin_memory: {pin_memory}")

    # Create model
    logger.info("Initializing model...")
    model = PolyphonyGCN(config)
    model = model.to(config.device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")

    # Create optimizer and schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = FocalPoissonNLLLoss(alpha=1.0, gamma=2.0)
    composer_criterion = FocalCrossEntropyLoss(alpha=1.0, gamma=2.0)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100),
        CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    ], milestones=[100])

    # Initialize TensorBoard logger
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "runs" / run_name
    tb_logger = TensorBoardLogger(log_dir, comment=f"_bs{batch_size}_lr{config.learning_rate}")
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    logger.info(f"View with: tensorboard --logdir={log_dir.parent}")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        composer_criterion=composer_criterion,
        device=config.device,
        config=config,
        tb_logger=tb_logger
    )

    # Load checkpoint if provided
    start_epoch = args.start_epoch
    if args.checkpoint:
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint_epoch, _ = trainer.load_checkpoint(
            args.checkpoint,
            map_location=config.device
        )
        if args.start_epoch == 0:
            start_epoch = checkpoint_epoch
        logger.info(f"Resuming from epoch {start_epoch}")

    # Start training with validation
    logger.info("Starting training...")
    validate_every_n_steps = getattr(config, 'validate_every_n_steps', 50)
    logger.info(f"Validation will run every {validate_every_n_steps} steps")
    trainer.train(train_loader, config.num_epochs, val_loader=val_loader, validate_every_n_steps=validate_every_n_steps, start_epoch=start_epoch)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
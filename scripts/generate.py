#!/usr/bin/env python3
"""
Music generation script using trained ASTGCN model.

This script loads a trained model and generates new music autoregressively
from a seed sequence.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.models.astgcn import PolyphonyGCN
from src.data.dataset import MidiGraphDataset
from configs.default_config import DefaultConfig


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    model = PolyphonyGCN(config).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

    return model


def get_seed_from_dataset(dataset, index=0):
    """Get a seed sequence from the dataset."""
    seed_data = dataset[index]

    # Convert to batch format (add batch dimension)
    seed_batch = {
        'features': seed_data['features'].unsqueeze(0),
        'feature_indices': seed_data['feature_indices'].unsqueeze(0),
        'input_graphs': seed_data.get('input_edge_index'),
        'target_graphs': seed_data.get('target_edge_index'),
        'global_graph': seed_data.get('global_graph'),
    }

    return seed_batch, seed_data


def generate_from_seed(
    model,
    seed_batch,
    num_steps=100,
    temperature=1.0,
    top_k=None,
    top_p=0.9
):
    """
    Generate music from a seed sequence.

    Args:
        model: Trained PolyphonyGCN model
        seed_batch: Seed sequence dictionary
        num_steps: Number of timesteps to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (if set)
        top_p: Nucleus sampling threshold

    Returns:
        generated: Dictionary with generated sequence
    """
    print(f"\nGenerating {num_steps} timesteps...")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k if top_k else 'None'}")
    print(f"Top-p: {top_p if top_p else 'None'}")

    with torch.no_grad():
        generated = model.generate(
            seed_sequence=seed_batch,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_graphs=False
        )

    print(f"Generated sequence shape: {generated['features'].shape}")
    return generated


def save_generated_sequence(generated, output_path):
    """Save generated sequence to file."""
    torch.save(generated, output_path)
    print(f"Saved generated sequence to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate music using trained ASTGCN model')

    # Model and data arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default=str(project_root / 'configs' / 'config.yml'),
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str,
                        default=str(project_root / 'data' / 'raw_test'),
                        help='Path to data directory for seed sequences')

    # Generation arguments
    parser.add_argument('--seed_index', type=int, default=0,
                        help='Index of seed sequence from dataset')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of timesteps to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling (if set)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling threshold')

    # Output arguments
    parser.add_argument('--output', type=str,
                        default=str(project_root / 'generated_music.pt'),
                        help='Output path for generated sequence')

    args = parser.parse_args()

    # Load configuration
    config = DefaultConfig(project_root=project_root, config_path=args.config)
    print(f"Using device: {config.device}")

    # Load model
    model = load_model(args.checkpoint, config)

    # Load dataset to get seed
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = MidiGraphDataset(
        npz_dir=Path(args.data_dir),
        seq_length=config.periods,
        time_step=config.time_step,
        max_pitch=127,
        config=config
    )

    if len(dataset) == 0:
        print("Error: No data found in dataset!")
        return

    print(f"Dataset contains {len(dataset)} segments")

    # Get seed sequence
    seed_batch, seed_data = get_seed_from_dataset(dataset, args.seed_index)
    print(f"\nUsing seed from index {args.seed_index}")
    print(f"Seed composer: {seed_data.get('composer_name', 'Unknown')}")
    print(f"Seed shape: {seed_batch['features'].shape}")

    # Generate
    generated = generate_from_seed(
        model=model,
        seed_batch=seed_batch,
        num_steps=args.num_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Save output
    save_generated_sequence(generated, args.output)

    print("\nGeneration complete!")
    print(f"To convert to MIDI, use: python scripts/export_midi.py --input {args.output}")


if __name__ == '__main__':
    main()
